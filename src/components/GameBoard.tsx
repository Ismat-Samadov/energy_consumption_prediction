'use client';
import { useState, useRef, useEffect, useCallback } from 'react';
import { Card as CardType, canPlaceOnFoundation, canPlaceOnTableau } from '@/lib/gameLogic';
import { Source, useKlondike } from '@/hooks/useKlondike';
import CardComp from './Card';

// ── Confetti ──────────────────────────────────────────────────────────────────
const CONFETTI_COLORS = ['#f5c842','#e74c3c','#3498db','#2ecc71','#9b59b6','#ff6b6b','#ffd93d'];

function Confetti() {
  const pieces = Array.from({ length: 60 }, (_, i) => ({
    id: i,
    color: CONFETTI_COLORS[i % CONFETTI_COLORS.length],
    left: `${Math.random() * 100}vw`,
    delay: `${Math.random() * 2}s`,
    duration: `${2.5 + Math.random() * 2}s`,
    size: `${6 + Math.random() * 6}px`,
    shape: i % 3 === 0 ? '50%' : i % 3 === 1 ? '0%' : '0%',
    rotate: i % 3 === 2 ? 'rotate(45deg)' : '',
  }));

  return (
    <>
      {pieces.map(p => (
        <div
          key={p.id}
          className="confetti"
          style={{
            left: p.left,
            top: '-12px',
            width: p.size,
            height: p.size,
            background: p.color,
            borderRadius: p.shape,
            transform: p.rotate,
            animationDuration: p.duration,
            animationDelay: p.delay,
          }}
        />
      ))}
    </>
  );
}

// ── Pile Slot ─────────────────────────────────────────────────────────────────
function PileSlot({
  hint, className = '', onDrop, onDragOver, onDragLeave, isOver, onClick,
}: {
  hint?: string;
  className?: string;
  onDrop?: (e: React.DragEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDragLeave?: () => void;
  isOver?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      className={`pile-slot ${isOver ? 'pile-slot-over' : ''} ${className}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={onClick}
    >
      {hint && (
        <span
          className="absolute inset-0 flex items-center justify-center select-none pointer-events-none text-white/25 font-bold"
          style={{ fontSize: 'clamp(16px, 3vw, 30px)' }}
        >
          {hint}
        </span>
      )}
    </div>
  );
}

// ── Stat Box ──────────────────────────────────────────────────────────────────
function StatBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col items-center bg-black/30 backdrop-blur-md rounded-2xl px-3 py-2 min-w-[54px] border border-white/10">
      <span className="text-white/45 font-semibold uppercase tracking-widest" style={{ fontSize: '8px' }}>
        {label}
      </span>
      <span className="text-white font-bold tabular-nums leading-tight" style={{ fontSize: 'clamp(12px,2vw,17px)' }}>
        {value}
      </span>
    </div>
  );
}

// ── Main Board ────────────────────────────────────────────────────────────────
export default function GameBoard() {
  const { game, won, score, moves, time, canUndo, newGame, draw, undo, move, autoFoundation } = useKlondike();

  const dragSrc   = useRef<Source | null>(null);
  const [overFoundation, setOverFoundation] = useState<number | null>(null);
  const [overTableau,    setOverTableau]    = useState<number | null>(null);
  const [selected,       setSelected]       = useState<Source | null>(null);

  // Clear selection on new game
  useEffect(() => { setSelected(null); }, [game.moves === 0 && game.stock.length === 24]);

  // ── Drag ──────────────────────────────────────────────────────────────────
  function handleDragStart(e: React.DragEvent, source: Source) {
    dragSrc.current = source;
    e.dataTransfer.effectAllowed = 'move';
    setSelected(null);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }

  function handleDropFoundation(e: React.DragEvent, pileIndex: number) {
    e.preventDefault();
    setOverFoundation(null);
    if (dragSrc.current) { move(dragSrc.current, 'foundation', pileIndex); dragSrc.current = null; }
  }

  function handleDropTableau(e: React.DragEvent, pileIndex: number) {
    e.preventDefault();
    setOverTableau(null);
    if (dragSrc.current) { move(dragSrc.current, 'tableau', pileIndex); dragSrc.current = null; }
  }

  // ── Click-to-select / click-to-move ──────────────────────────────────────
  const handleCardClick = useCallback((card: CardType, source: Source) => {
    if (!card.faceUp) return;

    if (!selected) {
      setSelected(source);
      return;
    }

    // Same card → deselect
    if (
      selected.area === source.area &&
      (selected.area === 'waste' || (selected as { pileIndex?: number }).pileIndex === (source as { pileIndex?: number }).pileIndex)
    ) {
      setSelected(null);
      return;
    }

    // Try to move selected → this position
    // Check if placing on tableau card position is valid
    if (source.area === 'tableau') {
      const pile = game.tableau[(source as { pileIndex: number }).pileIndex];
      const cardIndex = (source as { cardIndex: number }).cardIndex;
      // Can only drop onto the last card of a pile
      if (cardIndex === pile.length - 1) {
        const worked = tryMove(selected, 'tableau', (source as { pileIndex: number }).pileIndex);
        if (worked) { setSelected(null); return; }
      }
    }

    // Re-select the clicked card
    setSelected(source);
  }, [selected, game]);

  function tryMove(from: Source, toArea: 'foundation' | 'tableau', toPile: number): boolean {
    // Peek what cards we'd be moving
    let topCard: CardType | undefined;
    if (from.area === 'waste') topCard = game.waste[game.waste.length - 1];
    else if (from.area === 'foundation') topCard = game.foundations[(from as { pileIndex: number }).pileIndex].at(-1);
    else {
      const s = from as { pileIndex: number; cardIndex: number };
      topCard = game.tableau[s.pileIndex][s.cardIndex];
    }
    if (!topCard) return false;

    if (toArea === 'foundation') {
      const pile = game.foundations[toPile];
      if (!canPlaceOnFoundation(topCard, pile)) return false;
    } else {
      const pile = game.tableau[toPile];
      if (!canPlaceOnTableau(topCard, pile)) return false;
    }
    move(from, toArea, toPile);
    return true;
  }

  function handleSlotClick(toArea: 'foundation' | 'tableau', toPile: number) {
    if (!selected) return;
    const worked = tryMove(selected, toArea, toPile);
    if (worked) setSelected(null);
  }

  // ── Double-click → auto-foundation ───────────────────────────────────────
  function handleDoubleClick(card: CardType, source: Source) {
    setSelected(null);
    if (source.area === 'waste') {
      autoFoundation(card.id, 'waste');
    } else if (source.area === 'tableau') {
      autoFoundation(card.id, 'tableau', (source as { pileIndex: number }).pileIndex);
    }
  }

  // ── Card offsets ──────────────────────────────────────────────────────────
  const FD = 'clamp(12px, 2vw, 20px)';   // face-down
  const FU = 'clamp(20px, 3.3vw, 36px)'; // face-up

  function topValue(pile: CardType[], cardIdx: number) {
    if (cardIdx === 0) return 0;
    const parts = [];
    for (let k = 0; k < cardIdx; k++) parts.push(pile[k].faceUp ? FU : FD);
    return `calc(${parts.join(' + ')})`;
  }

  function isSelected(source: Source) {
    if (!selected) return false;
    if (selected.area !== source.area) return false;
    if (selected.area === 'waste') return true;
    const s = selected as { pileIndex: number; cardIndex?: number };
    const t = source  as { pileIndex: number; cardIndex?: number };
    if (s.pileIndex !== t.pileIndex) return false;
    if (selected.area === 'tableau') {
      return (selected as { cardIndex: number }).cardIndex <= (source as { cardIndex: number }).cardIndex;
    }
    return true;
  }

  const SUITS_ORDER = ['♠', '♥', '♣', '♦'];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="relative z-10 flex flex-col min-h-dvh p-2 sm:p-3 md:p-4 gap-2 sm:gap-3">

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between gap-2 flex-wrap">

        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-white/10 border border-white/20 flex items-center justify-center text-lg sm:text-xl backdrop-blur">
            ♠
          </div>
          <h1 className="text-white font-black tracking-tight hidden sm:block" style={{ fontSize: 'clamp(16px,2.5vw,24px)' }}>
            Klondike
          </h1>
        </div>

        {/* Stats */}
        <div className="flex gap-1.5 sm:gap-2">
          <StatBox label="Score" value={score} />
          <StatBox label="Time"  value={time}  />
          <StatBox label="Moves" value={moves} />
        </div>

        {/* Buttons */}
        <div className="flex gap-1.5 sm:gap-2">
          <button
            onClick={undo}
            disabled={!canUndo}
            title="Undo last move"
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 rounded-xl font-bold text-white border border-white/20 bg-white/10 hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed active:scale-95 transition-all backdrop-blur"
            style={{ fontSize: 'clamp(10px,1.5vw,13px)' }}
          >
            <span className="text-base leading-none">↩</span>
            <span className="hidden sm:inline">Undo</span>
          </button>
          <button
            onClick={newGame}
            title="Start a new game"
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 rounded-xl font-bold text-emerald-900 bg-yellow-400 hover:bg-yellow-300 shadow-lg active:scale-95 transition-all"
            style={{ fontSize: 'clamp(10px,1.5vw,13px)' }}
          >
            <span className="text-base leading-none">✦</span>
            <span className="hidden xs:inline">New</span>
            <span className="hidden sm:inline"> Game</span>
          </button>
        </div>
      </header>

      {/* ── TOP ROW ────────────────────────────────────────────────────────── */}
      <div className="flex items-start gap-1.5 sm:gap-2">

        {/* Stock */}
        <div className="flex-1 relative" style={{ maxWidth: 'calc(100% / 7)', aspectRatio: '2.5/3.5' }}>
          {game.stock.length > 0 ? (
            <div
              className="absolute inset-0 card-back cursor-pointer hover:brightness-110 active:scale-95 transition-all"
              onClick={draw}
              title="Draw card"
            >
              <span
                className="absolute inset-0 flex items-center justify-center font-bold text-white/50 select-none"
                style={{ fontSize: 'clamp(18px,3.5vw,34px)' }}
              >
                ↺
              </span>
            </div>
          ) : (
            <div
              className="absolute inset-0 pile-slot cursor-pointer hover:border-white/40 hover:bg-white/5 active:scale-95 transition-all flex flex-col items-center justify-center gap-0.5"
              onClick={draw}
              title="Reset stock"
            >
              <span className="text-white/35 font-bold" style={{ fontSize: 'clamp(18px,3.5vw,32px)' }}>↺</span>
              <span className="text-white/20 font-semibold" style={{ fontSize: '8px' }}>RESET</span>
            </div>
          )}
        </div>

        {/* Waste */}
        <div className="flex-1 relative" style={{ maxWidth: 'calc(100% / 7)', aspectRatio: '2.5/3.5' }}>
          {game.waste.length === 0 ? (
            <PileSlot />
          ) : (
            <div className="absolute inset-0">
              {game.waste.slice(-3).map((card, i, arr) => {
                const isLast = i === arr.length - 1;
                return (
                  <CardComp
                    key={card.id}
                    card={card}
                    source={{ area: 'waste' }}
                    onDragStart={isLast ? handleDragStart : () => {}}
                    onDoubleClick={isLast ? handleDoubleClick : undefined}
                    onClick={isLast ? handleCardClick : undefined}
                    isTop={isLast}
                    selected={isLast && isSelected({ area: 'waste' })}
                    style={{
                      left: `${i * 12}px`,
                      zIndex: i + 1,
                      width: '100%',
                    }}
                  />
                );
              })}
            </div>
          )}
        </div>

        {/* Spacer */}
        <div className="flex-1" style={{ maxWidth: 'calc(100% / 7)' }} />

        {/* Foundations */}
        {game.foundations.map((pile, i) => {
          const top = pile[pile.length - 1];
          return (
            <div key={i} className="flex-1 relative" style={{ maxWidth: 'calc(100% / 7)', aspectRatio: '2.5/3.5' }}>
              <PileSlot
                hint={SUITS_ORDER[i]}
                isOver={overFoundation === i}
                className="absolute inset-0"
                onDrop={e => handleDropFoundation(e, i)}
                onDragOver={e => { handleDragOver(e); setOverFoundation(i); }}
                onDragLeave={() => setOverFoundation(null)}
                onClick={() => handleSlotClick('foundation', i)}
              />
              {top && (
                <CardComp
                  card={top}
                  source={{ area: 'foundation', pileIndex: i }}
                  onDragStart={handleDragStart}
                  onDoubleClick={handleDoubleClick}
                  onClick={handleCardClick}
                  isTop
                  selected={isSelected({ area: 'foundation', pileIndex: i })}
                  style={{ zIndex: 2 }}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* ── TABLEAU ────────────────────────────────────────────────────────── */}
      <div className="flex gap-1.5 sm:gap-2 flex-1 pb-4">
        {game.tableau.map((pile, colIdx) => (
          <div
            key={colIdx}
            className="flex-1 relative"
            style={{ minHeight: '120px' }}
          >
            {/* Drop slot (always behind cards) */}
            <PileSlot
              hint="K"
              isOver={overTableau === colIdx}
              className="absolute inset-x-0 top-0"
              onDrop={e => handleDropTableau(e, colIdx)}
              onDragOver={e => { handleDragOver(e); setOverTableau(colIdx); }}
              onDragLeave={() => setOverTableau(null)}
              onClick={() => handleSlotClick('tableau', colIdx)}
            />

            {/* Cards container — grows to fit all stacked cards */}
            <div className="relative" style={{ width: '100%', aspectRatio: '2.5/3.5' }}>
              {pile.map((card, cardIdx) => {
                const src: Source = { area: 'tableau', pileIndex: colIdx, cardIndex: cardIdx };
                const isTop = cardIdx === pile.length - 1;
                return (
                  <CardComp
                    key={card.id}
                    card={card}
                    source={src}
                    onDragStart={card.faceUp ? handleDragStart : () => {}}
                    onDoubleClick={isTop ? handleDoubleClick : undefined}
                    onClick={card.faceUp ? handleCardClick : undefined}
                    isTop={isTop}
                    selected={card.faceUp && isSelected(src)}
                    onDrop={e => handleDropTableau(e, colIdx)}
                    onDragOver={e => { handleDragOver(e); setOverTableau(colIdx); }}
                    onDragLeave={() => setOverTableau(null)}
                    style={{
                      top: topValue(pile, cardIdx),
                      zIndex: cardIdx + 1,
                      width: '100%',
                      position: 'absolute',
                    }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* ── WIN OVERLAY ────────────────────────────────────────────────────── */}
      {won && (
        <>
          <Confetti />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/55 backdrop-blur-sm">
            <div className="win-card bg-gradient-to-br from-emerald-800 via-emerald-900 to-emerald-950 rounded-3xl p-7 sm:p-10 text-center shadow-2xl border border-emerald-500/25 w-full max-w-xs sm:max-w-sm">

              <div className="text-5xl sm:text-6xl mb-3">🃏</div>
              <h2 className="text-2xl sm:text-3xl font-black text-yellow-400 mb-1">You Win!</h2>
              <p className="text-white/60 text-sm mb-6">Excellent game!</p>

              <div className="grid grid-cols-3 gap-2 mb-6">
                {[
                  { label: 'Score', value: score },
                  { label: 'Time',  value: time  },
                  { label: 'Moves', value: moves },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-black/30 rounded-2xl py-3 border border-white/5">
                    <div className="text-white/45 text-xs uppercase tracking-widest mb-1">{label}</div>
                    <div className="text-white font-bold text-base sm:text-lg">{value}</div>
                  </div>
                ))}
              </div>

              <button
                onClick={newGame}
                className="btn-shimmer w-full py-3.5 rounded-2xl font-black text-emerald-900 text-base sm:text-lg shadow-xl hover:scale-[1.03] active:scale-[0.98] transition-transform"
              >
                Play Again
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
