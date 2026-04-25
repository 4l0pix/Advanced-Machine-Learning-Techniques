import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { fetchBranches, fetchFileTree, fetchFileContent } from './utils/github';
import Dock from './components/Dock';
import FileTree from './components/FileTree';
import ContentViewer from './components/ContentViewer';
import { CloudLightning, Github } from './components/Icons';

function App() {
  const [branches, setBranches] = useState([]);
  const [currentBranch, setCurrentBranch] = useState(null);
  const [tree, setTree] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initial fetch of branches
  useEffect(() => {
    const init = async () => {
      try {
        const repoBranches = await fetchBranches();
        setBranches(repoBranches);
        if (repoBranches.length > 0) {
          // Set primary branch (usually main or the first available)
          const primary = repoBranches.find(b => b.name === 'main') || repoBranches[0];
          handleBranchChange(primary.name);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    init();

    // "Worker" logic - refresh branches every 5 minutes
    const interval = setInterval(async () => {
      try {
        const repoBranches = await fetchBranches();
        setBranches(repoBranches);
      } catch (err) {
        console.error('Background background update failed:', err);
      }
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  const handleBranchChange = useCallback(async (branchName) => {
    setCurrentBranch(branchName);
    setLoading(true);
    try {
      const { tree } = await fetchFileTree(branchName);
      setTree(tree);
      
      // Try to find README.md or the first file to show initially
      const readme = tree.find(file => file.path.toLowerCase() === 'readme.md');
      if (readme) {
        handleFileSelect(readme.path, branchName);
      } else if (tree.length > 0) {
        handleFileSelect(tree[0].path, branchName);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFileSelect = useCallback(async (path, branchName = currentBranch) => {
    setSelectedFile(path);
    setLoading(true);
    try {
      const fileContent = await fetchFileContent(path, branchName);
      setContent(fileContent);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [currentBranch]);

  return (
    <div className="app-container">
      <header className="px-8 py-4 glass border-b border-base02 flex justify-between items-center animate-in slide-in-from-top duration-500">
        <div className="flex items-center gap-3">
          <CloudLightning className="text-cyan animate-pulse" size={28} />
          <div>
            <h1 className="text-2xl font-bold m-0 text-base2 leading-tight">Advanced Machine Learning Techniques</h1>
            <p className="text-sm text-base01 m-0">Project Repository Explorer • {currentBranch || 'Loading...'}</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <a href="https://github.com/4l0pix/Advanced-Machine-Learning-Techniques" target="_blank" rel="noreferrer" className="flex items-center gap-2 hover:text-cyan transition-colors">
            <Github size={20} />
            <span>View Source</span>
          </a>
        </div>
      </header>

      <main className="main-content">
        <FileTree 
          tree={tree} 
          selectedFile={selectedFile} 
          onFileSelect={handleFileSelect} 
        />
        
        <ContentViewer 
          content={content} 
          fileName={selectedFile || ''} 
          loading={loading} 
        />
      </main>

      <Dock 
        branches={branches} 
        currentBranch={currentBranch} 
        onSelect={handleBranchChange} 
      />
      
      {error && (
        <div className="fixed top-4 right-4 bg-red text-white p-4 rounded-lg shadow-lg z-50">
          Error: {error}
        </div>
      )}
    </div>
  );
}

export default App;
