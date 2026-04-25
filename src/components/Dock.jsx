import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { FolderGit2 } from './Icons';

const DISTANCE = 150;
const SIZE = 48;
const ENLARGED_SIZE = 80;

function DockIcon({ branch, active, onClick, mouseX }) {
  const ref = useRef(null);
  
  const distance = useTransform(mouseX, (val) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  const widthSpring = useSpring(useTransform(distance, [-DISTANCE, 0, DISTANCE], [SIZE, ENLARGED_SIZE, SIZE]), {
    mass: 0.1,
    stiffness: 150,
    damping: 12,
  });

  return (
    <motion.div
      ref={ref}
      style={{ width: widthSpring, height: widthSpring }}
      onClick={() => onClick(branch.name)}
      className={`dock-item glass ${active ? 'active' : ''}`}
    >
      <FolderGit2 size={24} />
      <span className="dock-tooltip">{branch.name}</span>
    </motion.div>
  );
}

export default function Dock({ branches, currentBranch, onSelect }) {
  const mouseX = useMotionValue(Infinity);

  return (
    <div className="dock-wrapper">
      <motion.div
        onMouseMove={(e) => mouseX.set(e.pageX)}
        onMouseLeave={() => mouseX.set(Infinity)}
        className="dock-container glass"
      >
        {branches.map((branch) => (
          <DockIcon
            key={branch.name}
            branch={branch}
            active={currentBranch === branch.name}
            onClick={onSelect}
            mouseX={mouseX}
          />
        ))}
      </motion.div>
    </div>
  );
}
