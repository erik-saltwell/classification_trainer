from __future__ import annotations

from pathlib import Path

from classification_trainer.console.file_logging_protocol import FileLogger


# ---------------------------------------------------------------------------
# US1: Core logging methods
# ---------------------------------------------------------------------------


def test_report_message(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_message("hello")
    content = log.read_text()
    assert "hello" in content


def test_report_warning(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_warning("caution")
    content = log.read_text()
    assert "WARNING" in content
    assert "caution" in content


def test_report_error(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_error("failure")
    content = log.read_text()
    assert "ERROR" in content
    assert "failure" in content


def test_report_exception(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    try:
        raise ValueError("oops")
    except ValueError as exc:
        logger.report_exception("ctx", exc)
    content = log.read_text()
    assert "EXCEPTION" in content
    assert "ctx" in content
    assert "ValueError" in content
    assert "oops" in content


def test_report_table_message(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_table_message({"name": "alice", "age": "30"})
    content = log.read_text()
    assert "name" in content
    assert "alice" in content
    assert "age" in content
    assert "30" in content


def test_report_multicolumn_table(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_multicolumn_table(["col_a", "col_b"], [["x", "y"], ["1", "2"]])
    content = log.read_text()
    assert "col_a" in content
    assert "col_b" in content
    assert "x" in content
    assert "y" in content


def test_add_break(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_message("before")
    logger.add_break(3)
    logger.report_message("after")
    content = log.read_text()
    # 3 blank lines between "before" and "after"
    assert "before" in content
    assert "after" in content
    lines = content.split("\n")
    # Find consecutive empty lines
    empty_count = 0
    for line in lines:
        if line.strip() == "":
            empty_count += 1
    assert empty_count >= 3


def test_status_produces_no_output(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    with logger.status("working...") as handle:
        handle.update("still working...")
    content = log.read_text()
    assert content.strip() == ""


def test_progress_produces_no_output(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    with logger.progress("processing", total=100) as task:
        task.advance(50)
        task.set_description("halfway")
    content = log.read_text()
    assert content.strip() == ""


# ---------------------------------------------------------------------------
# US2: Overwrite on creation
# ---------------------------------------------------------------------------


def test_overwrites_existing_file(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    log.write_text("old content that should be gone")
    logger = FileLogger(log)
    logger.report_message("new content")
    content = log.read_text()
    assert "old content" not in content
    assert "new content" in content


def test_second_logger_overwrites_first(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger1 = FileLogger(log)
    logger1.report_message("first run")
    logger2 = FileLogger(log)
    logger2.report_message("second run")
    content = log.read_text()
    assert "first run" not in content
    assert "second run" in content


# ---------------------------------------------------------------------------
# US3: Strip Rich markup
# ---------------------------------------------------------------------------


def test_strips_blue_markup(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_message("[blue]Loading model...[/blue]")
    content = log.read_text()
    assert "Loading model..." in content
    assert "[blue]" not in content
    assert "[/blue]" not in content


def test_strips_red_markup_in_error(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.report_error("[red]Error:[/red] something failed")
    content = log.read_text()
    assert "Error:" in content
    assert "something failed" in content
    assert "[red]" not in content
    assert "[/red]" not in content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_creates_parent_directories(tmp_path: Path) -> None:
    log = tmp_path / "sub" / "dir" / "log.txt"
    logger = FileLogger(log)
    logger.report_message("nested")
    assert log.exists()
    assert "nested" in log.read_text()


def test_add_break_zero(tmp_path: Path) -> None:
    log = tmp_path / "out.txt"
    logger = FileLogger(log)
    logger.add_break(0)
    content = log.read_text()
    assert content.strip() == ""
