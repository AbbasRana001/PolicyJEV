import type { ReactNode } from "react";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  children: ReactNode;
  id?: string;
}

/** An on/off switch with a text label, as in the design. */
export function Switch({ checked, onChange, children, id }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      id={id}
      className="sw"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
    >
      <span className="t" aria-hidden="true" />
      {children}
    </button>
  );
}
