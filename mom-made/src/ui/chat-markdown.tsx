import React, { memo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const plugins = [remarkGfm];

export const ChatMarkdown = memo(function ChatMarkdown({
  id,
  text,
  openLink,
}: {
  id: string;
  text: string;
  openLink: (url: string) => void;
}) {
  const prefix = `chat-${id}-`;
  return (
    <Markdown
      skipHtml
      remarkPlugins={plugins}
      remarkRehypeOptions={{ clobberPrefix: prefix }}
      components={{
        a: ({ node: _node, href, children, ...props }) => {
          if (href?.startsWith("#" + prefix)) {
            return (
              <a
                {...props}
                href={href}
                onClick={(event) => {
                  event.preventDefault();
                  document
                    .getElementById(href.slice(1))
                    ?.scrollIntoView({ block: "nearest" });
                }}
              >
                {children}
              </a>
            );
          }
          if (!href || !/^https?:\/\//i.test(href)) return <>{children}</>;
          return (
            <a
              {...props}
              href={href}
              onClick={(event) => {
                event.preventDefault();
                openLink(href);
              }}
            >
              {children}
            </a>
          );
        },
        img: ({ alt }) => <>{alt}</>,
        table: ({ node: _node, ...props }) => (
          <div className="markdown-table">
            <table {...props} />
          </div>
        ),
      }}
    >
      {text}
    </Markdown>
  );
});
