# ♠ Klondike Solitaire

A polished, fully-featured Klondike Solitaire game built with **Next.js 16**, **TypeScript**, and **Tailwind CSS v4**. Fully responsive — works great on desktop, tablet, and mobile.

---

## Features

- **Classic Klondike rules** — draw 1 card, build foundations A→K by suit, alternate colors on tableau
- **Drag & drop** — native HTML5 drag-and-drop between all piles
- **Click to select / click to place** — tap a card to select it, tap a valid destination to move it (great for mobile)
- **Double-click** — instantly sends a card to the correct foundation
- **Undo** — step back through your full move history
- **Score system** — earn points for smart moves, lose points for recycling the stock
- **Timer & move counter** — track your performance every game
- **Win celebration** — confetti animation + stats summary on completion
- **Responsive design** — adapts from large monitors down to 320px wide phones
- **Green felt aesthetic** — realistic card design with gloss, shadows, and texture

---

## Scoring

| Action | Points |
|---|---|
| Waste → Tableau | +5 |
| Waste → Foundation | +10 |
| Tableau → Foundation | +10 |
| Flip a card face-up | +5 |
| Foundation → Tableau | −15 |
| Recycle stock | −100 |
| Win bonus | +1000 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Next.js 16](https://nextjs.org) (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS v4 |
| State | React `useReducer` + custom hook |
| Drag & Drop | HTML5 native DnD API |

---

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

```bash
# Build for production
npm run build
npm start
```

---

## Project Structure

```
src/
├── app/
│   ├── globals.css       # Green felt theme, card styles, animations
│   ├── layout.tsx        # Root layout + metadata + favicon
│   └── page.tsx          # Entry point → renders GameBoard
├── components/
│   ├── Card.tsx          # Card face / back rendering, drag & drop, click handlers
│   └── GameBoard.tsx     # Full game UI — header, piles, win overlay, confetti
├── hooks/
│   └── useKlondike.ts    # Game state (useReducer), timer, undo history
└── lib/
    └── gameLogic.ts      # Pure functions: deck, shuffle, move validation, scoring
```

---

## How to Play

1. **Stock** (top-left): Click to draw a card onto the waste pile. When empty, click to recycle.
2. **Waste** (next to stock): The top card is available to play.
3. **Foundations** (top-right, 4 piles): Build each suit from **A up to K**. Filling all four wins!
4. **Tableau** (7 columns): Stack cards in **alternating colors, descending rank**. Only **Kings** can start an empty column.

### Controls

| Action | How |
|---|---|
| Move a card | Drag and drop |
| Select a card (mobile) | Single tap |
| Place selected card | Tap destination |
| Auto-send to foundation | Double-click top card |
| Undo last move | Click **↩ Undo** |
| Start over | Click **✦ New Game** |

---

## License

MIT — free to use and modify.
