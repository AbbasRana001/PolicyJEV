import type { ReactNode } from "react";

import type { ResultsPayload } from "../api";
import { DataLoading, DataNotAvailable } from "./DataNotAvailable";

/** One piece of the results payload, or why it cannot be shown. */
export type Part<T> =
  | { state: "loading" }
  | { state: "failed" }
  | { state: "missing" }
  | { state: "ok"; data: T };

/** Pick a section of the results; undefined means the request is still running, null that it failed. */
export function part<T>(
  results: ResultsPayload | null | undefined,
  pick: (payload: ResultsPayload) => T | undefined,
): Part<T> {
  if (results === undefined) return { state: "loading" };
  if (results === null) return { state: "failed" };
  const data = pick(results);
  return data === undefined ? { state: "missing" } : { state: "ok", data };
}

/** Render the data, or a clear "Data not available" box. */
export function renderPart<T>(value: Part<T>, render: (data: T) => ReactNode): ReactNode {
  switch (value.state) {
    case "loading":
      return <DataLoading />;
    case "failed":
      return <DataNotAvailable detail="Start the API with python -m policyjev.api, then reload this page." />;
    case "missing":
      return <DataNotAvailable />;
    case "ok":
      return render(value.data);
  }
}
