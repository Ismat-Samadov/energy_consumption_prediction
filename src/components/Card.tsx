'use client';
import { Card as CardType, isRed } from '@/lib/gameLogic';
import { Source } from '@/hooks/useKlondike';

interface CardProps {
  card: CardType;
  style?: React.CSSProperties;
  source: Source;
  onDragStart: (e: React.DragEvent, source: Source) => void;
  onDoubleClick?: (card: CardType, source: Source) => void;
  onDrop?: (e: React.DragEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDragLeave?: () => void;
  onClick?: (card: CardType, source: Source) => void;
  isTop?: boolean;
  selected?: boolean;
}

const FACE_LABEL: Record<string, string> = { J: 'J', Q: 'Q', K: 'K' };

export default function Card({
  card, style, source, onDragStart, onDoubleClick,
  onDrop, onDragOver, onDragLeave, onClick, isTop, selected,
}: CardProps) {
  const red   = isRed(card.suit);
  const color = red ? '#c0392b' : '#1a1a2e';
  const isFace = card.rank in FACE_LABEL;

  if (!card.faceUp) {
    return (
      <div
        className="absolute w-full card-back"
        style={{ aspectRatio: '2.5/3.5', ...style }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      />
    );
  }

  return (
    <div
      className={`absolute w-full card-face ${selected ? 'card-selected' : ''}`}
      style={{ aspectRatio: '2.5/3.5', ...style }}
      draggable
      onDragStart={e => onDragStart(e, source)}
      onDoubleClick={isTop ? () => onDoubleClick?.(card, source) : undefined}
      onClick={() => onClick?.(card, source)}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      {/* Top-left corner */}
      <div
        className="absolute flex flex-col items-center leading-none z-10"
        style={{ top: '5%', left: '7%', color, gap: '1px' }}
      >
        <span className="font-black" style={{ fontSize: 'clamp(8px, 1.6vw, 15px)', lineHeight: 1 }}>
          {card.rank}
        </span>
        <span style={{ fontSize: 'clamp(6px, 1.2vw, 11px)', lineHeight: 1 }}>
          {card.suit}
        </span>
      </div>

      {/* Center */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-[1]">
        {isFace ? (
          <div className="flex flex-col items-center" style={{ gap: '2px' }}>
            <span
              className="font-black font-serif"
              style={{ fontSize: 'clamp(16px, 3.2vw, 34px)', color, opacity: 0.65, lineHeight: 1 }}
            >
              {card.rank}
            </span>
            <span style={{ fontSize: 'clamp(12px, 2.2vw, 22px)', color, opacity: 0.45, lineHeight: 1 }}>
              {card.suit}
            </span>
          </div>
        ) : (
          <span
            style={{
              fontSize: card.rank === 'A'
                ? 'clamp(24px, 4.5vw, 48px)'
                : 'clamp(18px, 3.5vw, 38px)',
              color,
              opacity: card.rank === 'A' ? 0.75 : 0.1,
              lineHeight: 1,
            }}
          >
            {card.suit}
          </span>
        )}
      </div>

      {/* Bottom-right corner (rotated) */}
      <div
        className="absolute flex flex-col items-center leading-none rotate-180 z-10"
        style={{ bottom: '5%', right: '7%', color, gap: '1px' }}
      >
        <span className="font-black" style={{ fontSize: 'clamp(8px, 1.6vw, 15px)', lineHeight: 1 }}>
          {card.rank}
        </span>
        <span style={{ fontSize: 'clamp(6px, 1.2vw, 11px)', lineHeight: 1 }}>
          {card.suit}
        </span>
      </div>
    </div>
  );
}
