import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function pct(n: number): number {
  if (!isFinite(n) || n <= 0) return 0;
  return Math.round(Math.min(1, n) * 100);
}
