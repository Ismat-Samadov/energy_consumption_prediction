'use client';
import { useCallback, useEffect, useReducer, useRef } from 'react';
import {
  GameState, Card,
  newGame, canPlaceOnFoundation, canPlaceOnTableau, isGameWon, SCORE,
} from '@/lib/gameLogic';

// ── Types ─────────────────────────────────────────────────────────────────────

export type Source =
  | { area: 'waste' }
  | { area: 'foundation'; pileIndex: number }
  | { area: 'tableau'; pileIndex: number; cardIndex: number };

interface KlondikeState {
  game: GameState;
  history: GameState[];
  won: boolean;
  seconds: number;
  running: boolean;
}

type Action =
  | { type: 'NEW_GAME' }
  | { type: 'DRAW' }
  | { type: 'MOVE'; from: Source; toArea: 'foundation' | 'tableau'; toPile: number }
  | { type: 'AUTO_FOUNDATION'; cardId: string; fromArea: 'waste' | 'tableau'; fromPile?: number }
  | { type: 'UNDO' }
  | { type: 'TICK' };

// ── Reducer ───────────────────────────────────────────────────────────────────

function cloneGame(g: GameState): GameState {
  return {
    stock: g.stock.map(c => ({ ...c })),
    waste: g.waste.map(c => ({ ...c })),
    foundations: g.foundations.map(f => f.map(c => ({ ...c }))),
    tableau: g.tableau.map(t => t.map(c => ({ ...c }))),
    score: g.score,
    moves: g.moves,
  };
}

function pickCards(from: Source, game: GameState): Card[] | null {
  if (from.area === 'waste') {
    const top = game.waste[game.waste.length - 1];
    return top ? [top] : null;
  }
  if (from.area === 'foundation') {
    const top = game.foundations[from.pileIndex].at(-1);
    return top ? [top] : null;
  }
  // tableau
  const pile = game.tableau[from.pileIndex];
  return pile.slice(from.cardIndex);
}

function removeCards(from: Source, game: GameState) {
  if (from.area === 'waste') {
    game.waste.pop();
  } else if (from.area === 'foundation') {
    game.foundations[from.pileIndex].pop();
  } else {
    game.tableau[from.pileIndex].splice(from.cardIndex);
    // flip top card
    const pile = game.tableau[from.pileIndex];
    if (pile.length > 0 && !pile[pile.length - 1].faceUp) {
      pile[pile.length - 1].faceUp = true;
      game.score += SCORE.TURN_OVER;
    }
  }
}

function reducer(state: KlondikeState, action: Action): KlondikeState {
  switch (action.type) {
    case 'NEW_GAME':
      return {
        game: newGame(),
        history: [],
        won: false,
        seconds: 0,
        running: true,
      };

    case 'TICK':
      return state.running ? { ...state, seconds: state.seconds + 1 } : state;

    case 'UNDO': {
      if (state.history.length === 0) return state;
      const prev = state.history[state.history.length - 1];
      return {
        ...state,
        game: prev,
        history: state.history.slice(0, -1),
        won: false,
      };
    }

    case 'DRAW': {
      const g = cloneGame(state.game);
      if (g.stock.length === 0) {
        // recycle waste back to stock
        g.stock = g.waste.reverse().map(c => ({ ...c, faceUp: false }));
        g.waste = [];
        g.score = Math.max(0, g.score - 100);
      } else {
        const card = g.stock.pop()!;
        card.faceUp = true;
        g.waste.push(card);
      }
      g.moves += 1;
      return { ...state, game: g, history: [...state.history, state.game] };
    }

    case 'MOVE': {
      const { from, toArea, toPile } = action;
      const g = cloneGame(state.game);
      const cards = pickCards(from, g);
      if (!cards || cards.length === 0) return state;

      if (toArea === 'foundation') {
        if (cards.length !== 1) return state;
        if (!canPlaceOnFoundation(cards[0], g.foundations[toPile])) return state;
        removeCards(from, g);
        g.foundations[toPile].push(cards[0]);
        g.score += from.area === 'waste' ? SCORE.WASTE_TO_FOUNDATION : SCORE.TABLEAU_TO_FOUNDATION;
        if (from.area === 'foundation') g.score += SCORE.FOUNDATION_TO_TABLEAU; // penalty
      } else {
        if (!canPlaceOnTableau(cards[0], g.tableau[toPile])) return state;
        removeCards(from, g);
        g.tableau[toPile].push(...cards);
        if (from.area === 'waste') g.score += SCORE.WASTE_TO_TABLEAU;
        if (from.area === 'foundation') g.score += SCORE.FOUNDATION_TO_TABLEAU;
      }
      g.moves += 1;
      const won = isGameWon(g);
      if (won) g.score += 1000;
      return {
        ...state,
        game: g,
        history: [...state.history, state.game],
        won,
        running: won ? false : state.running,
      };
    }

    case 'AUTO_FOUNDATION': {
      const { cardId, fromArea, fromPile } = action;
      const g = cloneGame(state.game);

      let card: Card | undefined;
      let from: Source;

      if (fromArea === 'waste') {
        card = g.waste.find(c => c.id === cardId);
        from = { area: 'waste' };
      } else {
        const pIdx = fromPile!;
        card = g.tableau[pIdx].find(c => c.id === cardId);
        from = { area: 'tableau', pileIndex: pIdx, cardIndex: g.tableau[pIdx].findIndex(c => c.id === cardId) };
      }

      if (!card) return state;
      const destIdx = g.foundations.findIndex(f => canPlaceOnFoundation(card!, f));
      if (destIdx === -1) return state;

      removeCards(from, g);
      g.foundations[destIdx].push(card);
      g.score += fromArea === 'waste' ? SCORE.WASTE_TO_FOUNDATION : SCORE.TABLEAU_TO_FOUNDATION;
      g.moves += 1;

      const won = isGameWon(g);
      if (won) g.score += 1000;
      return {
        ...state,
        game: g,
        history: [...state.history, state.game],
        won,
        running: won ? false : state.running,
      };
    }

    default:
      return state;
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

const INITIAL: KlondikeState = {
  game: newGame(),
  history: [],
  won: false,
  seconds: 0,
  running: true,
};

export function useKlondike() {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer
  useEffect(() => {
    if (state.running) {
      intervalRef.current = setInterval(() => dispatch({ type: 'TICK' }), 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [state.running]);

  const newGame = useCallback(() => dispatch({ type: 'NEW_GAME' }), []);
  const draw    = useCallback(() => dispatch({ type: 'DRAW' }), []);
  const undo    = useCallback(() => dispatch({ type: 'UNDO' }), []);

  const move = useCallback((from: Source, toArea: 'foundation' | 'tableau', toPile: number) => {
    dispatch({ type: 'MOVE', from, toArea, toPile });
  }, []);

  const autoFoundation = useCallback((cardId: string, fromArea: 'waste' | 'tableau', fromPile?: number) => {
    dispatch({ type: 'AUTO_FOUNDATION', cardId, fromArea, fromPile });
  }, []);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  return {
    game: state.game,
    won: state.won,
    score: state.game.score,
    moves: state.game.moves,
    time: formatTime(state.seconds),
    canUndo: state.history.length > 0,
    newGame,
    draw,
    undo,
    move,
    autoFoundation,
  };
}
