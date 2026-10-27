"""Shared exception types for pdfcancel conversions."""

from __future__ import annotations


class ConversionIncomplete(Exception):
    """A document could not be fully transcribed this run.

    Completed pages are always checkpointed to the page cache, so re-running
    resumes from where it stopped. The attributes distinguish a clean
    interruption (a SIGINT/SIGTERM during a long run) from individual pages
    that failed after exhausting retries.
    """

    def __init__(
        self,
        *,
        total: int,
        completed: int,
        failed_pages: list[int] | None = None,
        interrupted: bool = False,
    ) -> None:
        self.total = total
        self.completed = completed
        self.failed_pages = failed_pages or []
        self.interrupted = interrupted
        if interrupted:
            msg = f"interrupted after {completed}/{total} page(s); re-run to resume"
        else:
            shown = ", ".join(str(p) for p in self.failed_pages[:10])
            more = " …" if len(self.failed_pages) > 10 else ""
            msg = (
                f"{completed}/{total} page(s) done; "
                f"{len(self.failed_pages)} page(s) failed [{shown}{more}]; "
                "re-run to retry the missing pages"
            )
        super().__init__(msg)
