# Dark Mode Design System — App Shell

> Used for: Dashboard, sidebar, search interface, agent progress panel, navigation
> Overrides MASTER.md colors and typography for the app shell context

---

**Project:** DeepResearch v2
**Mode:** Dark Mode (OLED)
**Performance:** Excellent | **Accessibility:** WCAG AAA

---

## Color Palette

| Role | Hex | CSS Variable | Tailwind |
|------|-----|--------------|----------|
| Background | `#0F172A` | `--color-bg-dark` | `slate-950` |
| Surface | `#1E293B` | `--color-surface-dark` | `slate-800` |
| Border | `#334155` | `--color-border-dark` | `slate-700` |
| CTA/Accent | `#22C55E` | `--color-cta-dark` | `green-500` |
| Agent Accent | `#8B5CF6` | `--color-agent-dark` | `violet-500` |
| Text Primary | `#F8FAFC` | `--color-text-dark` | `slate-50` |
| Text Secondary | `#94A3B8` | `--color-text-muted-dark` | `slate-400` |
| Text Tertiary | `#64748B` | `--color-text-dim-dark` | `slate-500` |
| Error | `#EF4444` | `--color-error-dark` | `red-500` |
| Warning | `#F59E0B` | `--color-warning-dark` | `amber-500` |
| Info | `#3B82F6` | `--color-info-dark` | `blue-500` |

---

## Typography

- **Heading Font:** Exo (weights: 300, 400, 500, 600, 700)
- **Data/Mono Font:** Roboto Mono (weights: 300, 400, 500, 700)
- **Mood:** Science, technology, research, data, futuristic, precise

**Google Fonts:**
```
https://fonts.google.com/share?selection.family=Exo:wght@300;400;500;600;700|Roboto+Mono:wght@300;400;500;700
```

**CSS Import:**
```css
@import url('https://fonts.googleapis.com/css2?family=Exo:wght@300;400;500;600;700&family=Roboto+Mono:wght@300;400;500;700&display=swap');
```

---

## Component Specs

### Buttons (Dark Mode)

```css
.btn-primary-dark {
  background: #22C55E;
  color: #0F172A;
  padding: 12px 24px;
  border-radius: 8px;
  font-family: 'Exo', sans-serif;
  font-weight: 600;
  transition: all 200ms ease;
  cursor: pointer;
}

.btn-primary-dark:hover {
  background: #16A34A;
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.3);
}

.btn-secondary-dark {
  background: transparent;
  color: #8B5CF6;
  border: 1px solid #8B5CF6;
  padding: 12px 24px;
  border-radius: 8px;
  font-family: 'Exo', sans-serif;
  font-weight: 600;
  transition: all 200ms ease;
  cursor: pointer;
}

.btn-secondary-dark:hover {
  background: rgba(139, 92, 246, 0.1);
}
```

### Cards (Dark Mode)

```css
.card-dark {
  background: #1E293B;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 24px;
  transition: all 200ms ease;
  cursor: pointer;
}

.card-dark:hover {
  border-color: #8B5CF6;
  box-shadow: 0 0 20px rgba(139, 92, 246, 0.1);
}
```

### Inputs (Dark Mode)

```css
.input-dark {
  background: #0F172A;
  color: #F8FAFC;
  padding: 12px 16px;
  border: 1px solid #334155;
  border-radius: 8px;
  font-size: 16px;
  font-family: 'Exo', sans-serif;
  transition: border-color 200ms ease;
}

.input-dark:focus {
  border-color: #8B5CF6;
  outline: none;
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
}

.input-dark::placeholder {
  color: #64748B;
}
```

### Sidebar (Dark Mode)

```css
.sidebar-dark {
  background: #0F172A;
  border-right: 1px solid #1E293B;
  width: 260px;
  padding: 16px;
}

.sidebar-item-dark {
  color: #94A3B8;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 150ms ease;
}

.sidebar-item-dark:hover {
  background: #1E293B;
  color: #F8FAFC;
}

.sidebar-item-dark.active {
  background: rgba(139, 92, 246, 0.15);
  color: #A78BFA;
}
```

---

## Glow Effects

```css
/* Minimal text glow for headings */
.glow-text {
  text-shadow: 0 0 10px rgba(139, 92, 246, 0.3);
}

/* Status indicator glow */
.glow-status-active {
  box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
}

/* Agent progress glow */
.glow-agent {
  box-shadow: 0 0 12px rgba(139, 92, 246, 0.2);
}
```

---

## Shadow Depths (Dark Mode)

| Level | Value | Usage |
|-------|-------|-------|
| `--shadow-dark-sm` | `0 1px 3px rgba(0,0,0,0.4)` | Subtle lift |
| `--shadow-dark-md` | `0 4px 8px rgba(0,0,0,0.5)` | Cards, dropdowns |
| `--shadow-dark-lg` | `0 10px 20px rgba(0,0,0,0.6)` | Modals, popovers |

---

## Anti-Patterns (Dark Mode Specific)

- Do NOT use pure black (`#000000`) — use `#0F172A` for depth
- Do NOT use white text on dark backgrounds below 4.5:1 contrast
- Do NOT use colored text on colored backgrounds without checking contrast
- Do NOT use bright saturated colors for large areas — use muted tones
- Do NOT animate glow effects continuously — use only on interaction
