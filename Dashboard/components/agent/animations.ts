import { Variants } from "framer-motion";

export const spawnVariants: Variants = {
  hidden: {
    scale: 0.8,
    opacity: 0,
    y: 20,
    filter: "blur(8px)",
  },
  visible: {
    scale: 1,
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: {
      type: "spring",
      stiffness: 180,
      damping: 22,
      mass: 1,
      duration: 0.6,
    },
  },
  exit: {
    scale: 0.9,
    opacity: 0,
    y: -10,
    filter: "blur(4px)",
    transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] },
  },
};

export const verticalLineVariants: Variants = {
  hidden: { scaleY: 0, opacity: 0 },
  visible: {
    scaleY: 1,
    opacity: 1,
    transition: {
      duration: 0.5,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
  },
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.1,
    },
  },
};

// New elegant cascade animation for tree nodes
export const cascadeVariants: Variants = {
  hidden: {
    opacity: 0,
    scale: 0.85,
    y: 30,
  },
  visible: (custom: number = 0) => ({
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 150,
      damping: 20,
      delay: custom * 0.12,
    },
  }),
};

// Connector line animation
export const lineGrowVariants: Variants = {
  hidden: {
    scaleY: 0,
    opacity: 0,
  },
  visible: (custom: number = 0) => ({
    scaleY: 1,
    opacity: 1,
    transition: {
      duration: 0.4,
      delay: custom * 0.08,
      ease: [0.22, 1, 0.36, 1],
    },
  }),
};

// Horizontal connector animation
export const horizontalLineVariants: Variants = {
  hidden: {
    scaleX: 0,
    opacity: 0,
  },
  visible: (custom: number = 0) => ({
    scaleX: 1,
    opacity: 1,
    transition: {
      duration: 0.35,
      delay: custom * 0.06,
      ease: [0.22, 1, 0.36, 1],
    },
  }),
};
