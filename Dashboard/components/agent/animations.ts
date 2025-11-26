import { Variants } from 'framer-motion';

export const spawnVariants: Variants = {
  hidden: { scale: 0.8, opacity: 0, y: 10 },
  visible: { 
    scale: 1, 
    opacity: 1, 
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 25 }
  },
  exit: { scale: 0.8, opacity: 0, transition: { duration: 0.2 } }
};

export const verticalLineVariants: Variants = {
  hidden: { scaleY: 0, opacity: 0 },
  visible: { 
    scaleY: 1, 
    opacity: 1, 
    transition: { duration: 0.5, ease: "easeInOut" }
  }
};