"""
Two-session unique-constraint conflicts: orioledb must behave like heap.

Every test plays the same script: s1 writes a row and keeps its transaction
open, s2 writes a conflicting row and has to wait for s1, s1 commits or rolls
back, s2 finishes.  The cases vary the isolation level, the unique index the
rows collide on, when s2 takes its snapshot and how s1 ends.

Each case runs twice in the same instance, on a heap table and on an orioledb
table, and both runs must agree: every statement returns the same rows or
SQLSTATE, s2's conflicting statement blocks in both or in neither, and the
table ends up the same.  The heap run is also checked against the PostgreSQL
rules for the scenario, so a broken scenario cannot pass by doing the same
wrong thing on both tables.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from difflib import SequenceMatcher
from typing import Any, Literal, NamedTuple

from testgres.connection import NodeConnection, pglib

from .base_test import BaseTest, ThreadQueryExecutor

AccessMethod = Literal['heap', 'orioledb']
Level = Literal['RC', 'RR']
UniqueIndex = Literal['pk', 'uniq']
# When s2 takes its snapshot.  after_s1_end is the baseline without
# concurrency: s2 starts only after s1 has ended.
S2Snapshot = Literal['before_s1_write', 'after_s1_write', 'after_s1_end']
S1End = Literal['COMMIT', 'ROLLBACK']

# (k, u, v): a row of table t.
Row = tuple[int, int, str]

# (statement, SQL, result) for every statement of both sessions, in order.
# A result is the rows returned (RETURNING k, SELECT), the command tag if
# there are none ('ROLLBACK' for a COMMIT of a failed transaction), or
# 'ERROR <SQLSTATE>'.
Results = list[tuple[str, str, str]]

ACCESS_METHODS: tuple[AccessMethod, ...] = ('heap', 'orioledb')
LEVELS: dict[Level, str] = {'RC': 'READ COMMITTED', 'RR': 'REPEATABLE READ'}
S2_SNAPSHOTS: tuple[S2Snapshot, ...] = ('before_s1_write', 'after_s1_write',
                                        'after_s1_end')
S1_ENDS: tuple[S1End, ...] = ('COMMIT', 'ROLLBACK')

SNAPSHOT_SQL = 'SELECT count(*) FROM t'
POLL_INTERVAL = 0.01


class Conflict(NamedTuple):
	"""Rows of s1 and s2 that collide on exactly one unique index."""
	s1_row: Row
	s2_row: Row
	target: str  # ON CONFLICT target: the column of that index
	pred: str  # selects the row s2 conflicts with


CONFLICTS: dict[UniqueIndex, Conflict] = {
    'pk': Conflict((1, 10, '1'), (1, 20, '2'), 'k', 'k = 1'),
    'uniq': Conflict((1, 10, '1'), (2, 10, '2'), 'u', 'u = 10'),
}


class Case(NamedTuple):
	level: Level
	unique: UniqueIndex  # the index the rows collide on
	s2_snapshot: S2Snapshot
	s1_end: S1End

	@property
	def conflict(self) -> Conflict:
		return CONFLICTS[self.unique]

	@property
	def begin(self) -> str:
		return f'BEGIN ISOLATION LEVEL {LEVELS[self.level]}'

	@property
	def concurrent(self) -> bool:
		"""s2 conflicts with s1's write while s1 is still open."""
		return self.s2_snapshot != 'after_s1_end'

	@property
	def s1_row_hidden(self) -> bool:
		"""s1 commits a row that s2's transaction snapshot does not see."""
		return (self.s1_end == 'COMMIT' and self.level == 'RR'
		        and self.concurrent)


class Outcome(NamedTuple):
	"""What one run of a case did; heap and orioledb must agree on it."""
	blocked: bool  # s2's conflicting statement had to wait for s1
	results: Results
	rows: list[Row]  # final contents of t


class UniqueConflictTest(BaseTest):
	ctl: NodeConnection

	def setUp(self):
		super().setUp()
		self.node.start()
		self.ctl = self.node.connect(autocommit=True)
		self.ctl.execute("SET statement_timeout = '10s'")
		self.ctl.execute('CREATE EXTENSION orioledb')

	def tearDown(self):
		try:
			self.ctl.close()
		finally:
			super().tearDown()

	def test_insert_vs_insert(self):
		for case in all_cases():
			with self.subTest(**case._asdict()):
				c = case.conflict
				s1_sql = insert_sql(c.s1_row)
				s2_sql = insert_sql(c.s2_row)
				outcomes: dict[AccessMethod, Outcome] = {}
				for am in ACCESS_METHODS:
					self._create_table(am)
					s1, s2 = self._connect(), self._connect()
					s1_pid, s2_pid = s1.pid, s2.pid
					results: Results = []
					blocked = False
					step(results, 's1_begin', s1, case.begin)
					if case.s2_snapshot == 'after_s1_end':
						step(results, 's1_write', s1, s1_sql)
						step(results, 's1_end', s1, case.s1_end)
						step(results, 's2_begin', s2, case.begin)
						step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's2_write', s2, s2_sql)
					else:
						step(results, 's2_begin', s2, case.begin)
						if case.s2_snapshot == 'before_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's1_write', s1, s1_sql)
						if case.s2_snapshot == 'after_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						write = ThreadQueryExecutor(s2, s2_sql)
						write.start()
						blocked = self._wait_blocked(s2_pid, s1_pid, write)
						step(results, 's1_end', s1, case.s1_end)
						join_step(results, 's2_write', s2, s2_sql, write)
					step(results, 's2_read', s2, read_sql(c))
					step(results, 's2_commit', s2, 'COMMIT')
					s1.close()
					s2.close()
					outcomes[am] = Outcome(blocked, results,
					                       self._final_rows(am))

				heap = outcomes['heap']
				if case.s1_end == 'COMMIT':
					want = 'ERROR 23505'
				else:
					want = returning(c.s2_row[0])
				self.assertHeapFollowsPg(heap, case, {'s2_write': want})
				self.assertSameAsHeap(heap, outcomes['orioledb'])

	def test_insert_vs_do_nothing(self):
		for case in all_cases():
			with self.subTest(**case._asdict()):
				c = case.conflict
				s1_sql = insert_sql(c.s1_row)
				s2_sql = insert_sql(c.s2_row, do_nothing(c))
				outcomes: dict[AccessMethod, Outcome] = {}
				for am in ACCESS_METHODS:
					self._create_table(am)
					s1, s2 = self._connect(), self._connect()
					s1_pid, s2_pid = s1.pid, s2.pid
					results: Results = []
					blocked = False
					step(results, 's1_begin', s1, case.begin)
					if case.s2_snapshot == 'after_s1_end':
						step(results, 's1_write', s1, s1_sql)
						step(results, 's1_end', s1, case.s1_end)
						step(results, 's2_begin', s2, case.begin)
						step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's2_write', s2, s2_sql)
					else:
						step(results, 's2_begin', s2, case.begin)
						if case.s2_snapshot == 'before_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's1_write', s1, s1_sql)
						if case.s2_snapshot == 'after_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						write = ThreadQueryExecutor(s2, s2_sql)
						write.start()
						blocked = self._wait_blocked(s2_pid, s1_pid, write)
						step(results, 's1_end', s1, case.s1_end)
						join_step(results, 's2_write', s2, s2_sql, write)
					step(results, 's2_read', s2, read_sql(c))
					step(results, 's2_commit', s2, 'COMMIT')
					s1.close()
					s2.close()
					outcomes[am] = Outcome(blocked, results,
					                       self._final_rows(am))

				heap = outcomes['heap']
				if case.s1_row_hidden:
					want = 'ERROR 40001'
				elif case.s1_end == 'COMMIT':
					want = returning()
				else:
					want = returning(c.s2_row[0])
				self.assertHeapFollowsPg(heap, case, {'s2_write': want})
				self.assertSameAsHeap(heap, outcomes['orioledb'])

	def test_insert_vs_do_update(self):
		for case in all_cases():
			with self.subTest(**case._asdict()):
				c = case.conflict
				s1_sql = insert_sql(c.s1_row)
				s2_sql = insert_sql(c.s2_row, do_update(c, c.s2_row))
				outcomes: dict[AccessMethod, Outcome] = {}
				for am in ACCESS_METHODS:
					self._create_table(am)
					s1, s2 = self._connect(), self._connect()
					s1_pid, s2_pid = s1.pid, s2.pid
					results: Results = []
					blocked = False
					step(results, 's1_begin', s1, case.begin)
					if case.s2_snapshot == 'after_s1_end':
						step(results, 's1_write', s1, s1_sql)
						step(results, 's1_end', s1, case.s1_end)
						step(results, 's2_begin', s2, case.begin)
						step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's2_write', s2, s2_sql)
					else:
						step(results, 's2_begin', s2, case.begin)
						if case.s2_snapshot == 'before_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's1_write', s1, s1_sql)
						if case.s2_snapshot == 'after_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						write = ThreadQueryExecutor(s2, s2_sql)
						write.start()
						blocked = self._wait_blocked(s2_pid, s1_pid, write)
						step(results, 's1_end', s1, case.s1_end)
						join_step(results, 's2_write', s2, s2_sql, write)
					step(results, 's2_read', s2, read_sql(c))
					step(results, 's2_commit', s2, 'COMMIT')
					s1.close()
					s2.close()
					outcomes[am] = Outcome(blocked, results,
					                       self._final_rows(am))

				heap = outcomes['heap']
				if case.s1_row_hidden:
					want = 'ERROR 40001'
				elif case.s1_end == 'COMMIT':
					want = returning(c.s1_row[0])
				else:
					want = returning(c.s2_row[0])
				self.assertHeapFollowsPg(heap, case, {'s2_write': want})
				self.assertSameAsHeap(heap, outcomes['orioledb'])

	def test_do_update_vs_do_update(self):
		for case in all_cases():
			with self.subTest(**case._asdict()):
				c = case.conflict
				s1_sql = insert_sql(c.s1_row, do_update(c, c.s1_row))
				s2_sql = insert_sql(c.s2_row, do_update(c, c.s2_row))
				outcomes: dict[AccessMethod, Outcome] = {}
				for am in ACCESS_METHODS:
					self._create_table(am)
					s1, s2 = self._connect(), self._connect()
					s1_pid, s2_pid = s1.pid, s2.pid
					results: Results = []
					blocked = False
					step(results, 's1_begin', s1, case.begin)
					if case.s2_snapshot == 'after_s1_end':
						step(results, 's1_write', s1, s1_sql)
						step(results, 's1_end', s1, case.s1_end)
						step(results, 's2_begin', s2, case.begin)
						step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's2_write', s2, s2_sql)
					else:
						step(results, 's2_begin', s2, case.begin)
						if case.s2_snapshot == 'before_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's1_write', s1, s1_sql)
						if case.s2_snapshot == 'after_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						write = ThreadQueryExecutor(s2, s2_sql)
						write.start()
						blocked = self._wait_blocked(s2_pid, s1_pid, write)
						step(results, 's1_end', s1, case.s1_end)
						join_step(results, 's2_write', s2, s2_sql, write)
					step(results, 's2_read', s2, read_sql(c))
					step(results, 's2_commit', s2, 'COMMIT')
					s1.close()
					s2.close()
					outcomes[am] = Outcome(blocked, results,
					                       self._final_rows(am))

				heap = outcomes['heap']
				if case.s1_row_hidden:
					want = 'ERROR 40001'
				elif case.s1_end == 'COMMIT':
					want = returning(c.s1_row[0])
				else:
					want = returning(c.s2_row[0])
				self.assertHeapFollowsPg(heap, case, {'s2_write': want})
				self.assertSameAsHeap(heap, outcomes['orioledb'])

	def test_upsert(self):
		# s2 runs the jepsen list-append client's upsert: UPDATE, and if that
		# found no row, INSERT under a savepoint, falling back to UPDATE again
		# if the INSERT fails.
		for case in all_cases():
			with self.subTest(**case._asdict()):
				c = case.conflict
				s1_sql = insert_sql(c.s1_row)
				s2_insert = insert_sql(c.s2_row)
				s2_update = update_sql(c, c.s2_row)
				outcomes: dict[AccessMethod, Outcome] = {}
				for am in ACCESS_METHODS:
					self._create_table(am)
					s1, s2 = self._connect(), self._connect()
					s1_pid, s2_pid = s1.pid, s2.pid
					results: Results = []
					blocked = False
					step(results, 's1_begin', s1, case.begin)
					if case.s2_snapshot == 'after_s1_end':
						step(results, 's1_write', s1, s1_sql)
						step(results, 's1_end', s1, case.s1_end)
						step(results, 's2_begin', s2, case.begin)
						step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
					else:
						step(results, 's2_begin', s2, case.begin)
						if case.s2_snapshot == 'before_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
						step(results, 's1_write', s1, s1_sql)
						if case.s2_snapshot == 'after_s1_write':
							step(results, 's2_snapshot', s2, SNAPSHOT_SQL)
					if step(results, 's2_update', s2,
					        s2_update) == returning():
						step(results, 's2_savepoint', s2, 'SAVEPOINT upsert')
						if case.s2_snapshot == 'after_s1_end':
							inserted = step(results, 's2_insert', s2,
							                s2_insert)
						else:
							insert = ThreadQueryExecutor(s2, s2_insert)
							insert.start()
							blocked = self._wait_blocked(
							    s2_pid, s1_pid, insert)
							step(results, 's1_end', s1, case.s1_end)
							inserted = join_step(results, 's2_insert', s2,
							                     s2_insert, insert)
						if inserted.startswith('ERROR'):
							step(results, 's2_rollback_to', s2,
							     'ROLLBACK TO SAVEPOINT upsert')
							step(results, 's2_update_again', s2, s2_update)
						else:
							step(results, 's2_release', s2,
							     'RELEASE SAVEPOINT upsert')
					step(results, 's2_read', s2, read_sql(c))
					step(results, 's2_commit', s2, 'COMMIT')
					s1.close()
					s2.close()
					outcomes[am] = Outcome(blocked, results,
					                       self._final_rows(am))

				heap = outcomes['heap']
				if case.s1_end == 'COMMIT' and not case.concurrent:
					# s1's row is visible, so the first UPDATE finds it.
					want = {'s2_update': returning(c.s1_row[0])}
				elif case.s1_end == 'COMMIT':
					# The INSERT fails.  The second UPDATE sees s1's row only
					# under READ COMMITTED, with its fresh statement snapshot.
					again = returning(
					    c.s1_row[0]) if case.level == 'RC' else returning()
					want = {
					    's2_update': returning(),
					    's2_insert': 'ERROR 23505',
					    's2_update_again': again,
					}
				else:
					want = {
					    's2_update': returning(),
					    's2_insert': returning(c.s2_row[0]),
					}
				self.assertHeapFollowsPg(heap, case, want)
				self.assertSameAsHeap(heap, outcomes['orioledb'])

	def assertHeapFollowsPg(self, heap: Outcome, case: Case,
	                        expected: dict[str, str]) -> None:
		"""The scenario does what it is meant to: heap behaves as PG says."""
		self.assertEqual(heap.blocked, case.concurrent, 'heap: blocked')
		results = {label: result for label, _, result in heap.results}
		for label, want in expected.items():
			self.assertEqual(results.get(label), want, f'heap: {label}')

	def assertSameAsHeap(self, heap: Outcome, oriole: Outcome) -> None:
		if oriole != heap:
			self.fail('heap and orioledb differ\n' +
			          side_by_side(heap, oriole))

	def _create_table(self, am: AccessMethod) -> None:
		self.ctl.execute('DROP TABLE IF EXISTS t')
		self.ctl.execute(
		    f'CREATE TABLE t (k int PRIMARY KEY, u int, v text) USING {am}')
		self.ctl.execute('CREATE UNIQUE INDEX t_u ON t (u)')

	def _connect(self) -> NodeConnection:
		con = self.node.connect(autocommit=True)
		# Fail a statement that blocks unexpectedly (error 57014) instead of
		# letting it hang the test.
		con.execute("SET statement_timeout = '10s'")
		return con

	def _wait_blocked(self, waiter_pid: int, holder_pid: int,
	                  thread: ThreadQueryExecutor) -> bool:
		"""
		Waits until the statement running in thread either blocks on the
		holder session or finishes; returns whether it blocked.
		statement_timeout ends the statement, and so the wait, within 10 s.
		"""
		while thread.is_alive():
			if query(
			    self.ctl, f'SELECT {holder_pid} = '
			    f'ANY(pg_blocking_pids({waiter_pid}))')[0][0]:
				return True
			time.sleep(POLL_INTERVAL)
		return False

	def _final_rows(self, am: AccessMethod) -> list[Row]:
		check = self.node.connect(autocommit=True)
		try:
			rows: list[Row] = query(check,
			                        'SELECT k, u, v FROM t ORDER BY k, u')
			# Look for duplicates without trusting the indexes.
			check.execute('SET enable_indexscan = off')
			check.execute('SET enable_indexonlyscan = off')
			check.execute('SET enable_bitmapscan = off')
			dups = query(
			    check, "SELECT 'k', k FROM t GROUP BY k HAVING count(*) > 1 "
			    "UNION ALL "
			    "SELECT 'u', u FROM t GROUP BY u HAVING count(*) > 1")
		finally:
			check.close()
		self.assertEqual(dups, [], f'{am}: duplicate keys in {rows}')
		return rows


# Helpers used by the tests above.


def all_cases() -> Iterator[Case]:
	for level in LEVELS:
		for unique in CONFLICTS:
			for s2_snapshot in S2_SNAPSHOTS:
				for s1_end in S1_ENDS:
					yield Case(level, unique, s2_snapshot, s1_end)


def insert_sql(row: Row, on_conflict: str = '') -> str:
	k, u, v = row
	sql = f"INSERT INTO t (k, u, v) VALUES ({k}, {u}, '{v}')"
	if on_conflict:
		sql += ' ' + on_conflict
	return sql + ' RETURNING k'


def do_nothing(c: Conflict) -> str:
	return f'ON CONFLICT ({c.target}) DO NOTHING'


def do_update(c: Conflict, row: Row) -> str:
	return f"ON CONFLICT ({c.target}) DO UPDATE SET v = t.v || ',{row[2]}'"


def update_sql(c: Conflict, row: Row) -> str:
	return f"UPDATE t SET v = v || ',{row[2]}' WHERE {c.pred} RETURNING k"


def read_sql(c: Conflict) -> str:
	return f'SELECT k, u, v FROM t WHERE {c.pred}'


def returning(*keys: int) -> str:
	"""The result of a statement that returned these values of k."""
	return str([(k, ) for k in keys])


def step(results: Results, label: str, con: NodeConnection, sql: str) -> str:
	"""Executes sql and records its result under label."""
	return record(results, label, sql, con, lambda: con.execute(sql))


def join_step(results: Results, label: str, con: NodeConnection, sql: str,
              thread: ThreadQueryExecutor) -> str:
	"""Waits for sql, running in thread, and records its result."""
	return record(results, label, sql, con, thread.join)


def record(results: Results, label: str, sql: str, con: NodeConnection,
           execute: Callable[[], Any]) -> str:
	try:
		rows = execute()
	except pglib.Error as e:
		result = f"ERROR {getattr(e, 'pgcode', None)}"
	else:
		if rows is None:
			# psycopg2's command tag: 'BEGIN', 'ROLLBACK' for a failed
			# COMMIT, ...
			result = str(getattr(con.cursor, 'statusmessage', 'ok'))
		else:
			result = str(rows)
	results.append((label, sql, result))
	return result


def side_by_side(heap: Outcome, oriole: Outcome) -> str:
	"""
	Both runs as a table, one line per statement, '!' marking the lines that
	differ, followed by whether s2 blocked and the final rows.
	"""
	heap_steps = {label: (sql, res) for label, sql, res in heap.results}
	oriole_steps = {label: (sql, res) for label, sql, res in oriole.results}

	# The runs may take different branches (the upsert does), so merge the two
	# label sequences, keeping each in its own order.
	a = [label for label, _, _ in heap.results]
	b = [label for label, _, _ in oriole.results]
	labels: list[str] = []
	for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
		for label in a[i1:i2] + (b[j1:j2] if tag != 'equal' else []):
			if label not in labels:
				labels.append(label)

	table = [(' ', 'step', 'SQL', 'heap', 'orioledb')]
	for label in labels:
		heap_sql, heap_res = heap_steps.get(label, ('', '-'))
		oriole_sql, oriole_res = oriole_steps.get(label, ('', '-'))
		mark = ' ' if (heap_sql, heap_res) == (oriole_sql, oriole_res) else '!'
		table.append((mark, label, heap_sql
		              or oriole_sql, heap_res, oriole_res))
	widths = [max(len(cell) for cell in column) for column in zip(*table)]
	lines = [
	    '  '.join(cell.ljust(width)
	              for cell, width in zip(row, widths)).rstrip()
	    for row in table
	]

	for name, heap_value, oriole_value in [
	    ('blocked', heap.blocked, oriole.blocked),
	    ('final rows', heap.rows, oriole.rows),
	]:
		if heap_value == oriole_value:
			lines.append(f'  {name}: {heap_value}')
		else:
			lines.append(
			    f'! {name}: heap {heap_value}, orioledb {oriole_value}')
	return '\n'.join(lines)


def query(con: NodeConnection, sql: str) -> list[Any]:
	"""Rows of a statement that returns rows."""
	rows = con.execute(sql)
	assert rows is not None, sql
	return rows
