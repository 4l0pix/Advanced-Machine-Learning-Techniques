const BASE_URL = 'https://api.github.com/repos/4l0pix/Advanced-Machine-Learning-Techniques';

export const fetchBranches = async () => {
  const response = await fetch(`${BASE_URL}/branches`);
  if (!response.ok) throw new Error('Failed to fetch branches');
  return await response.json();
};

export const fetchFileTree = async (branchName) => {
  // Get the latest commit SHA for the branch first
  const branchResponse = await fetch(`${BASE_URL}/branches/${branchName}`);
  const branchData = await branchResponse.json();
  const treeSha = branchData.commit.sha;

  // Fetch the tree recursively
  const response = await fetch(`${BASE_URL}/git/trees/${treeSha}?recursive=1`);
  if (!response.ok) throw new Error('Failed to fetch tree');
  return await response.json();
};

export const fetchFileContent = async (path, branchName) => {
  const rawUrl = `https://raw.githubusercontent.com/4l0pix/Advanced-Machine-Learning-Techniques/${branchName}/${path}`;
  
  try {
    // 1. Try fetching directly from raw.githubusercontent.com first
    // This is the most reliable way to get raw text with correct UTF-8 encoding.
    const rawResponse = await fetch(rawUrl);
    if (rawResponse.ok) {
      return await rawResponse.text();
    }

    // 2. Fallback to GitHub API if raw URL fails
    const response = await fetch(`${BASE_URL}/contents/${path}?ref=${branchName}`);
    if (!response.ok) throw new Error('Failed to fetch file content from both sources');
    
    const data = await response.json();
    
    if (data.content) {
      // Decode Base64 while properly handling UTF-8 characters
      const binaryString = atob(data.content.replace(/\s/g, ''));
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return new TextDecoder('utf-8').decode(bytes);
    }
    
    throw new Error('No content found in API response');
  } catch (error) {
    console.error('Error fetching file content:', error);
    throw error;
  }
};
