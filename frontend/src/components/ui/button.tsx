"use client";

import React, { useRef, useEffect, MouseEvent } from "react";
import gsap from "gsap";
import Tippy from "@tippyjs/react";
import { cn } from "../../lib/utils";
import "animate.css";
import "../../styles/buttons.css";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "icon" | "danger" | "loading";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  tooltip?: string;
  isActive?: boolean;
}

export function Button({
  variant = "secondary",
  tooltip,
  isActive,
  children,
  className = "",
  onClick,
  onMouseEnter,
  onMouseLeave,
  ...props
}: ButtonProps) {
  const btnRef = useRef<HTMLButtonElement>(null);
  const iconRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (btnRef.current) {
      iconRef.current = btnRef.current.querySelector('svg');
    }
  }, [children]);

  const handleMouseMove = (e: MouseEvent<HTMLButtonElement>) => {
    if (variant !== "primary" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const btn = btnRef.current;
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    gsap.to(btn, { x: x * 0.15, y: y * 0.15, duration: 0.3, ease: "power2.out" });
  };

  const handleMouseLeave = (e: MouseEvent<HTMLButtonElement>) => {
    const btn = btnRef.current;
    if (!btn) return;
    if (variant === "primary" && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.to(btn, { x: 0, y: 0, duration: 0.5, ease: "elastic.out(1, 0.5)" });
    }
    
    onMouseLeave?.(e);
  };

  const handleMouseEnter = (e: MouseEvent<HTMLButtonElement>) => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        onMouseEnter?.(e);
        return;
    }

    if (variant === "primary" && iconRef.current) {
      gsap.to(iconRef.current, { rotation: "+=360", duration: 0.6, ease: "power2.out" });
    }
    
    if (variant === "icon" && iconRef.current) {
      gsap.fromTo(iconRef.current, { scale: 1 }, { scale: 1.2, duration: 0.4, ease: "bounce.out", yoyo: true, repeat: 1 });
    }
    
    if (variant === "danger" && btnRef.current) {
      btnRef.current.classList.add("animate__animated", "animate__shakeX");
      setTimeout(() => {
        btnRef.current?.classList.remove("animate__animated", "animate__shakeX");
      }, 1000);
    }
    
    onMouseEnter?.(e);
  };

  const handleClick = (e: MouseEvent<HTMLButtonElement>) => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        onClick?.(e);
        return;
    }

    if (variant === "primary") {
      const btn = btnRef.current;
      if (btn) {
        const ripple = document.createElement("span");
        const rect = btn.getBoundingClientRect();
        ripple.className = "tg-ripple";
        
        // ensure ripple starts small and centered on click
        ripple.style.width = `20px`;
        ripple.style.height = `20px`;
        ripple.style.left = `${e.clientX - rect.left - 10}px`;
        ripple.style.top = `${e.clientY - rect.top - 10}px`;
        
        btn.appendChild(ripple);
        gsap.to(ripple, { 
          scale: 15, 
          opacity: 0, 
          duration: 0.6,
          ease: "power2.out", 
          onComplete: () => ripple.remove() 
        });
      }
    }
    
    if (variant === "icon" && iconRef.current) {
      gsap.to(iconRef.current, { rotation: "+=360", duration: 0.5, ease: "power2.out" });
    }

    onClick?.(e);
  };

  useEffect(() => {
    if (variant === "loading" && btnRef.current && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
       const btn = btnRef.current;
       let scanline = btn.querySelector('.tg-scanline');
       if (!scanline) {
         scanline = document.createElement('div');
         scanline.className = 'tg-scanline';
         btn.appendChild(scanline);
       }
       
       const tl = gsap.timeline({ repeat: -1 });
       tl.fromTo(scanline, { left: "-20%" }, { left: "120%", duration: 1.5, ease: "power1.inOut" });
       
       return () => { tl.kill(); scanline?.remove(); };
    }
  }, [variant]);

  let variantClasses = "";
  switch (variant) {
    case "primary": variantClasses = "tg-btn-primary px-6 py-3 rounded-lg"; break;
    case "secondary": variantClasses = "tg-btn-secondary px-5 py-2.5 rounded-md focus:animate__animated focus:animate__pulse"; break;
    case "ghost": variantClasses = cn("tg-btn-ghost px-4 py-2 rounded-md", isActive && "tg-btn-ghost-active text-white"); break;
    case "icon": variantClasses = "tg-btn-icon flex items-center justify-center rounded-xl"; break;
    case "danger": variantClasses = "tg-btn-danger px-5 py-2.5 rounded-md"; break;
    case "loading": variantClasses = "tg-btn-loading px-6 py-3 rounded-lg"; break;
    default: variantClasses = "tg-btn-secondary px-5 py-2.5 rounded-md";
  }

  const buttonContent = (
    <button
      ref={btnRef}
      className={cn(
        "tg-btn-base relative inline-flex items-center justify-center",
        variantClasses,
        className
      )}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      onClick={handleClick}
      disabled={variant === "loading" || props.disabled}
      {...props}
    >
      {variant === "loading" ? (
         <>
           <span className="relative z-10 flex items-center gap-2">
             Analyzing
             <span className="flex gap-0.5">
               <span className="animate-bounce delay-75">.</span>
               <span className="animate-bounce delay-150">.</span>
               <span className="animate-bounce delay-300">.</span>
             </span>
           </span>
         </>
      ) : children}
    </button>
  );

  if (tooltip) {
    return (
      <Tippy content={tooltip} animation="shift-away" theme="translucent" placement="top">
        <div className="inline-block">{buttonContent}</div>
      </Tippy>
    );
  }

  return buttonContent;
}
