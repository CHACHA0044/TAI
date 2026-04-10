"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Menu, X, FileText, Image, Video, Home } from "lucide-react";

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

  useEffect(() => {
    const handler = () => {
      // Use a slightly higher threshold for better visual separation
      setScrolled(window.scrollY > 10);
    };
    
    // Initial check
    handler();
    
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <>
      <motion.nav
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className={`fixed top-4 left-1/2 -translate-x-1/2 z-[100] w-[calc(100%-2rem)] max-w-5xl transition-all duration-500 ease-in-out rounded-2xl border ${
          scrolled
            ? "bg-[#05050a]/60 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] border-white/10 py-2.5 supports-[backdrop-filter]:bg-[#05050a]/40"
            : "bg-transparent backdrop-blur-none border-transparent py-4 shadow-none"
        }`}
      >
        <div className="flex items-center justify-between px-5 py-3">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500/30 blur-lg rounded-full group-hover:bg-blue-500/50 transition-all duration-300" />
              <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Shield className="w-4.5 h-4.5 text-white" />
              </div>
            </div>
            <span className="text-sm font-bold tracking-tight">
              <span className="text-gradient">TruthGuard</span>
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
                  className={`relative flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                    active
                      ? "text-white"
                      : "text-white/50 hover:text-white/80 hover:bg-white/5"
                  }`}
                >
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-600/30 to-purple-600/30 border border-white/10"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                  <Icon className="relative w-3.5 h-3.5" />
                  <span className="relative">{label}</span>
                </Link>
              );
            })}
          </div>

          {/* CTA + Mobile button */}
          <div className="flex items-center gap-3">
            <Link
              href="/text"
              className="hidden md:inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90 transition-opacity shadow-[0_4px_15px_rgba(99,102,241,0.3)]"
            >
              Try Now
            </Link>
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden flex items-center justify-center w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 transition-all"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="fixed top-[4.5rem] left-4 right-4 z-40 rounded-2xl glass-strong border border-white/10 shadow-2xl md:hidden"
          >
            <div className="p-3 flex flex-col gap-1">
              {navItems.map(({ label, href, icon: Icon }) => {
                const active = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                      active
                        ? "bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-white/10"
                        : "text-white/60 hover:text-white hover:bg-white/5"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </Link>
                );
              })}
              <Link
                href="/text"
                className="mt-2 flex items-center justify-center px-4 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-blue-600 to-purple-600 text-white"
              >
                Try Now
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
