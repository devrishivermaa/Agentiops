import React from 'react';
import { motion } from 'framer-motion';
import { verticalLineVariants } from './animations';

export const BranchConnector: React.FC<{
  type: 'vertical' | 'horizontal-left' | 'horizontal-right' | 'horizontal-full';
  className?: string;
}> = ({ type, className = '' }) => {
  const baseClass = "absolute bg-zinc-700/50";
  
  if (type === 'vertical') {
    return (
      <motion.div 
        variants={verticalLineVariants}
        initial="hidden"
        animate="visible"
        className={`${baseClass} w-0.5 h-full left-1/2 -ml-px origin-top ${className}`}
      >
        <motion.div 
          className="absolute top-0 left-0 w-full h-[30%] bg-gradient-to-b from-blue-500 to-transparent opacity-50"
          animate={{ top: ['-30%', '100%'] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
      </motion.div>
    );
  }
  
  // Horizontal connectors use borders to create the "Tree" look
  return (
    <div className={`absolute top-0 h-full w-1/2 border-t-2 border-zinc-700/50 ${
      type.includes('left') ? 'right-1/2 rounded-tl-xl' : 
      type.includes('right') ? 'left-1/2 rounded-tr-xl' : ''
    } ${className}`} />
  );
};