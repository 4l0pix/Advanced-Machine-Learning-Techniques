import React from 'react';
import ReactMarkdown from 'react-markdown';

export default function ContentViewer({ content, fileName, loading }) {
  if (loading) {
    return (
      <div className="viewer-area glass flex items-center justify-center">
        <div className="animate-pulse text-xl text-base1">Loading repository content...</div>
      </div>
    );
  }

  if (!content) {
    return (
      <div className="viewer-area glass flex items-center justify-center">
        <div className="text-xl text-base01">Select a file to view content or select a branch from the dock.</div>
      </div>
    );
  }

  const isMarkdown = fileName.endsWith('.md');

  return (
    <div className="viewer-area glass">
      <div className="file-header mb-6 pb-2 border-b border-base02 flex justify-between items-center">
        <h2 className="text-xl font-bold text-cyan m-0">{fileName}</h2>
      </div>
      
      <div className="content-rendered">
        {isMarkdown ? (
          <div className="markdown-body">
            <ReactMarkdown>
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          <pre className="p-4 bg-base02 rounded-lg overflow-x-auto">
            <code className="text-sm font-mono text-base1 leading-relaxed">
              {content}
            </code>
          </pre>
        )}
      </div>
    </div>
  );
}
