"use client";

import { useEffect, useId, useState } from "react";

export function Mermaid({ chart }: { chart: string }) {
  const [svg, setSvg] = useState("");
  const [errored, setErrored] = useState(false);
  const rawId = useId();
  const id = `mmd-${rawId.replace(/[^a-zA-Z0-9]/g, "")}`;

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "neutral",
          securityLevel: "loose",
          fontFamily: "var(--font-sans)",
        });
        const { svg } = await mermaid.render(id, chart);
        if (active) setSvg(svg);
      } catch {
        if (active) setErrored(true);
      }
    })();
    return () => {
      active = false;
    };
  }, [chart, id]);

  if (errored) {
    return (
      <pre className="my-6 overflow-x-auto rounded-lg border border-border bg-subtle p-4 text-xs text-muted-foreground">
        {chart}
      </pre>
    );
  }

  if (!svg) {
    return <div className="my-6 h-40 animate-pulse rounded-lg bg-muted" />;
  }

  return (
    <div
      className="my-6 flex justify-center overflow-x-auto rounded-lg border border-border bg-card p-4 [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
