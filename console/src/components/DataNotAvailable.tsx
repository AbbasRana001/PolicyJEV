/** What a chart section shows when the API has no data for it. */
export function DataNotAvailable({ detail }: { detail?: string }) {
  return (
    <p className="na-box" data-state="unavailable">
      Data not available{detail === undefined ? "." : `. ${detail}`}
    </p>
  );
}

export function DataLoading() {
  return (
    <p className="na-box" data-state="loading">
      Loading.
    </p>
  );
}
