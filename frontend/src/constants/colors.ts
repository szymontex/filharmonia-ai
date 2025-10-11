/**
 * Shared color constants for audio classification categories
 */

export const CLASS_COLORS = {
  MUSIC: {
    bg: 'bg-blue-100',
    text: 'text-blue-800',
    rgba: 'rgba(96, 165, 250, 0.8)', // blue-400 - darker and more opaque
  },
  APPLAUSE: {
    bg: 'bg-green-100',
    text: 'text-green-800',
    rgba: 'rgba(74, 222, 128, 0.8)', // green-400 - darker and more opaque
  },
  SPEECH: {
    bg: 'bg-purple-100',
    text: 'text-purple-800',
    rgba: 'rgba(192, 132, 252, 0.8)', // purple-400 - darker and more opaque
  },
  PUBLIC: {
    bg: 'bg-yellow-100',
    text: 'text-yellow-800',
    rgba: 'rgba(250, 204, 21, 0.8)', // yellow-400 - darker and more opaque
  },
  TUNING: {
    bg: 'bg-orange-100',
    text: 'text-orange-800',
    rgba: 'rgba(251, 146, 60, 0.8)', // orange-400 - darker and more opaque
  },
} as const

export type ClassType = keyof typeof CLASS_COLORS
