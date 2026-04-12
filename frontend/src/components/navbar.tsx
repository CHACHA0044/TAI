"use client";

/* Breakpoints Addressed:
 * xs/sm: Mobile hamburger menu, modal drawer trapping, emerald glows, 44px tap targets.
 * md: Sticky top navigation, layout shifts to horizontal flex.
 * lg+ : Desktop wide layout, preserved max constraints.
 */

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { Shield, Menu, X, FileText, Image, Video, Home } from "lucide-react";
import { Button } from "./ui/button";

const navItems = [
  { label: "Home", href: "/", icon: Home },
  { label: "Text", href: "/text", icon: FileText },
  { label: "Image", href: "/image", icon: Image },
  { label: "Video", href: "/video", icon: Video },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const prefersReducedMotion = useReducedMotion();

  useEffect(() => {
    const handler = () => {
      setScrolled(window.scrollY > 10);
    };
    
    handler();
    
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Trap focus / block scroll when mobile menu is open
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
  }, [mobileOpen]);

  return (
    <>
      <motion.nav
        initial={prefersReducedMotion ? { opacity: 1 } : { y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className={`fixed top-4 left-1/2 -translate-x-1/2 z-[100] w-[calc(100%-2rem)] max-w-[1440px] transition-all duration-500 ease-in-out rounded-2xl border will-change-transform ${
          scrolled
            ? "bg-[#05050a]/80 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] border-white/10 py-2 sm:py-2.5 supports-[backdrop-filter]:bg-[#05050a]/60"
            : "bg-transparent backdrop-blur-none border-transparent py-3 sm:py-4 shadow-none"
        }`}
      >
        <div className="flex items-center justify-between px-4 sm:px-5">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center gap-2 sm:gap-2.5 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 rounded-lg p-1"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-emerald-500/30 blur-lg rounded-full group-hover:bg-emerald-500/50 transition-all duration-300" aria-hidden="true" />
              <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <Shield className="w-4.5 h-4.5 text-white" />
              </div>
            </div>
            <span className="text-sm font-bold tracking-tight">
              <span className="bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent transform-gpu">TruthGuard</span>
              <span className="text-white/60 font-medium"> AI</span>
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map(({ label, href, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`relative flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                    active
                      ? "text-white"
                      : "text-white/50 hover:text-white/80 hover:bg-white/5"
                  }`}
                >
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-xl bg-gradient-to-r from-emerald-600/30 to-teal-600/30 border border-white/10 pointer-events-none"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      aria-hidden="true"
                    />
                  )}
                  <Icon className="relative w-3.5 h-3.5" />
                  <span className="relative">{label}</span>
                </Link>
              );
            })}
          </div>

          {/* CTA + Mobile button */}
          <div className="flex items-center gap-2 sm:gap-3">
            <Button
              variant="primary"
              onClick={() => router.push('/text')}
              className="hidden md:inline-flex px-5 py-2 h-10 rounded-xl text-sm font-semibold"
            >
              Try Now
            </Button>
            <Button
              variant="icon"
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden w-11 h-11 sm:w-10 sm:h-10 rounded-lg text-white/70 hover:text-white"
              aria-label={mobileOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileOpen}
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="fixed top-[5rem] left-4 right-4 z-[90] rounded-2xl bg-[#05050a]/95 backdrop-blur-xl supports-[backdrop-filter]:bg-[#05050a]/80 border border-white/10 shadow-2xl md:hidden overflow-hidden"
            role="dialog"
            aria-modal="true"
          >
            <div className="p-4 flex flex-col gap-2">
              {navItems.map(({ label, href, icon: Icon }) => {
                const active = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-3 px-4 py-3.5 rounded-xl text-base font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                      active
                        ? "bg-gradient-to-r from-emerald-600/20 to-teal-600/20 text-white border border-emerald-500/20"
                        : "text-white/60 hover:text-white hover:bg-white/5 active:bg-white/10"
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${active ? "text-emerald-400" : ""}`} />
                    {label}
                  </Link>
                );
              })}
              <div className="mt-4 pt-4 border-t border-white/10">
                <Button
                  onClick={() => router.push('/text')}
                  variant="primary"
                  className="w-full py-4 rounded-xl text-base font-semibold"
                >
                  Try Now Open System
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
