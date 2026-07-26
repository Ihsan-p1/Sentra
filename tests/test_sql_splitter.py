"""
Unit tests for schema statement splitting.

Regression cover for a naive schema.split(';'), which tore the plpgsql trigger
function in database/schema.sql into fragments that all failed to execute.
"""
import os

from database.connection import split_sql_statements

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "schema.sql")


def test_simple_statements_are_split():
    schema = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);"
    assert len(split_sql_statements(schema)) == 2


def test_comments_and_blank_lines_do_not_create_empty_statements():
    schema = "-- a comment\n\nCREATE TABLE a (id INT);\n\n-- another\n"
    statements = split_sql_statements(schema)

    assert len(statements) == 2  # the table, plus the trailing comment block
    assert all(s.strip() for s in statements)


def test_dollar_quoted_function_body_stays_in_one_statement():
    schema = """
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""
    statements = split_sql_statements(schema)

    assert len(statements) == 1
    assert "BEGIN" in statements[0]
    assert "RETURN NEW;" in statements[0]
    assert statements[0].rstrip().endswith("LANGUAGE plpgsql;")


def _leading_keywords(statement):
    """
    The DDL a statement actually starts with, ignoring leading comment lines.

    Matching on `"CREATE TABLE" in statement` broke the moment a comment
    explained why CREATE TABLE IF NOT EXISTS cannot add a column - the prose was
    counted as a table. Statements are identified by what they execute.
    """
    for line in statement.splitlines():
        line = line.strip()
        if line and not line.startswith("--"):
            return line.upper()
    return ""


def test_real_schema_file_parses_into_executable_statements():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        statements = split_sql_statements(f.read())

    function_statements = [s for s in statements if "CREATE OR REPLACE FUNCTION" in s]

    assert len(function_statements) == 1
    assert function_statements[0].count("$$") == 2

    heads = [_leading_keywords(s) for s in statements]

    # Every table in the schema still gets its own statement
    assert sum(1 for h in heads if h.startswith("CREATE TABLE")) == 4
    # Column migrations are separate statements, not swallowed by a neighbour
    assert sum(1 for h in heads if h.startswith("ALTER TABLE")) == 3


def test_migrations_are_idempotent():
    """
    The schema is executed on every startup, so an ALTER that is not guarded by
    IF NOT EXISTS would raise on the second run. init_database swallows
    'already exists' errors, which would hide a real failure behind a benign one.
    """
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        statements = split_sql_statements(f.read())

    for statement in statements:
        head = _leading_keywords(statement)
        if head.startswith("ALTER TABLE"):
            assert "IF NOT EXISTS" in statement.upper(), statement
        if head.startswith(("CREATE TABLE", "CREATE INDEX")):
            assert "IF NOT EXISTS" in statement.upper(), statement
