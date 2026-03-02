export type Suit = '♠' | '♥' | '♦' | '♣';
export type Rank = 'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K';

export interface Card {
  id: string;
  suit: Suit;
  rank: Rank;
  faceUp: boolean;
}

export interface GameState {
  stock: Card[];
  waste: Card[];
  foundations: Card[][];   // 4 piles
  tableau: Card[][];       // 7 piles
  score: number;
  moves: number;
}

const SUITS: Suit[] = ['♠', '♥', '♦', '♣'];
const RANKS: Rank[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
const RANK_VALUE: Record<Rank, number> = {
  A:1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7,
  '8':8, '9':9, '10':10, J:11, Q:12, K:13,
};

export const rankValue = (r: Rank) => RANK_VALUE[r];
export const isRed = (s: Suit) => s === '♥' || s === '♦';

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function createDeck(): Card[] {
  return shuffle(
    SUITS.flatMap(suit =>
      RANKS.map(rank => ({ id: `${rank}${suit}`, suit, rank, faceUp: false }))
    )
  );
}

export function newGame(): GameState {
  const deck = createDeck();
  const tableau: Card[][] = Array.from({ length: 7 }, () => []);

  let idx = 0;
  for (let col = 0; col < 7; col++) {
    for (let row = 0; row <= col; row++) {
      const card = { ...deck[idx++] };
      card.faceUp = row === col;
      tableau[col].push(card);
    }
  }

  return {
    stock: deck.slice(idx).map(c => ({ ...c, faceUp: false })),
    waste: [],
    foundations: [[], [], [], []],
    tableau,
    score: 0,
    moves: 0,
  };
}

export function canPlaceOnFoundation(card: Card, pile: Card[]): boolean {
  if (pile.length === 0) return card.rank === 'A';
  const top = pile[pile.length - 1];
  return top.suit === card.suit && rankValue(card.rank) === rankValue(top.rank) + 1;
}

export function canPlaceOnTableau(card: Card, pile: Card[]): boolean {
  if (pile.length === 0) return card.rank === 'K';
  const top = pile[pile.length - 1];
  if (!top.faceUp) return false;
  return isRed(card.suit) !== isRed(top.suit) && rankValue(card.rank) === rankValue(top.rank) - 1;
}

export function isGameWon(state: GameState): boolean {
  return state.foundations.every(f => f.length === 13);
}

// Score helpers
export const SCORE = {
  WASTE_TO_TABLEAU: 5,
  WASTE_TO_FOUNDATION: 10,
  TABLEAU_TO_FOUNDATION: 10,
  TURN_OVER: 5,
  FOUNDATION_TO_TABLEAU: -15,
};
