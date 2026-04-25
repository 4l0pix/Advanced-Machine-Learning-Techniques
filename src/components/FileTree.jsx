import React from 'react';
import { File, Folder, CheckCircle2 } from './Icons';

export default function FileTree({ tree, selectedFile, onFileSelect }) {
  if (!tree || tree.length === 0) {
    return (
      <div className="sidebar glass flex flex-col items-center justify-center text-base01 italic">
        No files found in this branch.
      </div>
    );
  }

  // Filter for common ML/Development files
  const filteredTree = tree.filter(item => 
    item.type === 'blob' && 
    !item.path.includes('.git/') && 
    !item.path.includes('node_modules/')
  ).sort((a, b) => {
    // Put READMEs at the top
    if (a.path.toLowerCase().includes('readme.md')) return -1;
    if (b.path.toLowerCase().includes('readme.md')) return 1;
    return a.path.localeCompare(b.path);
  });

  return (
    <div className="sidebar glass animate-in slide-in-from-left duration-300">
      <div className="flex items-center gap-2 mb-6 pb-2 border-b border-base02">
        <Folder size={18} className="text-yellow" />
        <h3 className="m-0 text-lg text-base2 font-semibold">Explorer</h3>
      </div>
      
      <div className="file-list overflow-y-auto">
        {filteredTree.map((file) => (
          <div
            key={file.path}
            onClick={() => onFileSelect(file.path)}
            className={`flex items-center gap-2 py-2 px-3 rounded-md cursor-pointer transition-all hover:bg-base02 
              ${selectedFile === file.path ? 'bg-base02 text-cyan border-l-2 border-cyan pl-2' : 'text-base01'}`}
          >
            <File size={16} className={selectedFile === file.path ? 'text-cyan' : 'text-base01'} />
            <span className="truncate text-sm">{file.path}</span>
            {selectedFile === file.path && <CheckCircle2 size={12} className="ml-auto" />}
          </div>
        ))}
      </div>
    </div>
  );
}
