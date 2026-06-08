"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { Mermaid } from "./mermaid";

function nodeText(node: React.ReactNode): string {
  if (node == null) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (React.isValidElement(node)) {
    return nodeText((node.props as { children?: React.ReactNode }).children);
  }
  return "";
}

export function LessonMarkdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[
        rehypeSlug,
        rehypeRaw,
        [rehypeHighlight, { detect: true, ignoreMissing: true }],
      ]}
      components={{
        pre(props) {
          const child = (
            Array.isArray(props.children) ? props.children[0] : props.children
          ) as React.ReactElement<{
            className?: string;
            children?: React.ReactNode;
          }> | null;
          const cls = child?.props?.className ?? "";
          if (/language-mermaid/.test(cls)) {
            return <Mermaid chart={nodeText(child?.props?.children).trim()} />;
          }
          return <pre {...props} />;
        },
        a({ href, children, ...rest }) {
          const external = typeof href === "string" && href.startsWith("http");
          return (
            <a
              href={href}
              target={external ? "_blank" : undefined}
              rel={external ? "noreferrer" : undefined}
              {...rest}
            >
              {children}
            </a>
          );
        },
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
