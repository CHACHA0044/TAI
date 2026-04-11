"use client";

import React from "react";
import { Copy, Share, Download, RefreshCw, Trash2, Home, FileText, Image as ImageIcon, Video } from "lucide-react";
import { Button } from "./button";

export function ButtonDemo() {
  return (
    <div className="p-8 space-y-12 bg-[#080818] min-h-screen text-white font-sans">
      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">1. Primary Buttons</h2>
        <div className="flex gap-4 items-center">
          <Button variant="primary">Invoke Guardian</Button>
          <Button variant="primary">
            <RefreshCw className="w-5 h-5 mr-2 inline-block" />
            Try Now
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">2. Secondary Buttons</h2>
        <div className="flex gap-4 items-center">
          <Button variant="secondary">New Session</Button>
          <Button variant="secondary">URL Tab</Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">3. Ghost / Nav Buttons</h2>
        <div className="flex gap-2 items-center bg-black/40 p-4 rounded-xl border border-white/5">
          <Button variant="ghost" tooltip="Go to home" isActive={true}>
            <Home className="w-4 h-4 mr-2" /> Home
          </Button>
          <Button variant="ghost" tooltip="Ctrl+T">
            <FileText className="w-4 h-4 mr-2" /> Text
          </Button>
          <Button variant="ghost" tooltip="Ctrl+I">
            <ImageIcon className="w-4 h-4 mr-2" /> Image
          </Button>
          <Button variant="ghost" tooltip="Ctrl+V">
            <Video className="w-4 h-4 mr-2" /> Video
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">4. Icon Buttons</h2>
        <div className="flex gap-4 items-center">
          <Button variant="icon" tooltip="Copy URL">
            <Copy className="w-5 h-5" />
          </Button>
          <Button variant="icon" tooltip="Share Result">
            <Share className="w-5 h-5" />
          </Button>
          <Button variant="icon" tooltip="Download PDF">
            <Download className="w-5 h-5" />
          </Button>
          <Button variant="icon" tooltip="Reload Analysis">
            <RefreshCw className="w-5 h-5" />
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">5. Danger Buttons</h2>
        <div className="flex gap-4 items-center">
          <Button variant="danger" tooltip="Irreversible Action">
            <Trash2 className="w-5 h-5 mr-2 inline-block" />
            Delete Model
          </Button>
          <Button variant="danger">Clear Cache</Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-bold border-b border-white/10 pb-2">6. Loading State Button</h2>
        <div className="flex gap-4 items-center">
          <Button variant="loading">Analyzing...</Button>
        </div>
      </div>
    </div>
  );
}
