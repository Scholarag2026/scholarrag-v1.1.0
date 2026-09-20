import { useCallback, useReducer } from "react";

interface NavEntry {
  nodeId: string;
  title: string;
}

interface NavState {
  history: NavEntry[];
  currentIndex: number;
}

type NavAction =
  | { type: "push"; nodeId: string; title: string }
  | { type: "back" }
  | { type: "forward" }
  | { type: "jumpTo"; index: number };

function navReducer(state: NavState, action: NavAction): NavState {
  switch (action.type) {
    case "push": {
      const trimmed = state.history.slice(0, state.currentIndex + 1);
      return {
        history: [...trimmed, { nodeId: action.nodeId, title: action.title }],
        currentIndex: trimmed.length,
      };
    }
    case "back":
      if (state.currentIndex <= 0) return state;
      return { ...state, currentIndex: state.currentIndex - 1 };
    case "forward":
      if (state.currentIndex >= state.history.length - 1) return state;
      return { ...state, currentIndex: state.currentIndex + 1 };
    case "jumpTo": {
      const idx = Math.max(0, Math.min(action.index, state.history.length - 1));
      return { ...state, currentIndex: idx };
    }
  }
}

export interface GraphNavigation {
  history: NavEntry[];
  currentIndex: number;
  push: (nodeId: string, title: string) => void;
  goBack: () => NavEntry | null;
  goForward: () => NavEntry | null;
  jumpTo: (index: number) => NavEntry | null;
  canGoBack: boolean;
  canGoForward: boolean;
  current: NavEntry | null;
}

export function useGraphNavigation(): GraphNavigation {
  const [state, dispatch] = useReducer(navReducer, {
    history: [],
    currentIndex: -1,
  });

  const push = useCallback((nodeId: string, title: string) => {
    dispatch({ type: "push", nodeId, title });
  }, []);

  const goBack = useCallback((): NavEntry | null => {
    if (state.currentIndex <= 0) return null;
    dispatch({ type: "back" });
    return state.history[state.currentIndex - 1];
  }, [state]);

  const goForward = useCallback((): NavEntry | null => {
    if (state.currentIndex >= state.history.length - 1) return null;
    dispatch({ type: "forward" });
    return state.history[state.currentIndex + 1];
  }, [state]);

  const jumpTo = useCallback((index: number): NavEntry | null => {
    if (index < 0 || index >= state.history.length) return null;
    dispatch({ type: "jumpTo", index });
    return state.history[index];
  }, [state]);

  return {
    history: state.history,
    currentIndex: state.currentIndex,
    push,
    goBack,
    goForward,
    jumpTo,
    canGoBack: state.currentIndex > 0,
    canGoForward: state.currentIndex < state.history.length - 1,
    current: state.currentIndex >= 0 ? state.history[state.currentIndex] : null,
  };
}
