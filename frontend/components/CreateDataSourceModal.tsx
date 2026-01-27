"use client";

import { useState, useRef } from "react";

interface CreateDataSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: {
    name: string;
    type: string;
    serviceAccountJson?: File | null;
  }) => void;
}

export default function CreateDataSourceModal({
  isOpen,
  onClose,
  onSubmit,
}: CreateDataSourceModalProps) {
  const [name, setName] = useState("");
  const [type, setType] = useState("google_drive");
  const [serviceAccountFile, setServiceAccountFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate that it's a JSON file
      if (file.type === "application/json" || file.name.endsWith(".json")) {
        setServiceAccountFile(file);
      } else {
        alert("Please upload a JSON file");
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      alert("Please enter a data source name");
      return;
    }
    onSubmit({
      name: name.trim(),
      type,
      serviceAccountJson: serviceAccountFile,
    });
    // Reset form
    setName("");
    setType("google_drive");
    setServiceAccountFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleClose = () => {
    setName("");
    setType("google_drive");
    setServiceAccountFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="glass w-full max-w-md rounded-2xl p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Create Data Source</h2>
          <button
            onClick={handleClose}
            className="text-zinc-400 hover:text-white transition-colors"
          >
            <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Data Source Name */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-zinc-300 mb-1">
              Data Source Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full rounded-lg border border-white/10 bg-white/5 py-2 px-3 text-sm text-white placeholder-zinc-500 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
              placeholder="Enter data source name"
            />
          </div>

          {/* Data Source Type */}
          <div>
            <label htmlFor="type" className="block text-sm font-medium text-zinc-300 mb-1">
              Data Source Type <span className="text-red-400">*</span>
            </label>
            <select
              id="type"
              value={type}
              onChange={(e) => setType(e.target.value)}
              required
              className="w-full rounded-lg border border-white/10 bg-white/5 py-2 px-3 text-sm text-white focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
            >
              <option value="google_drive">Google Drive</option>
              <option value="google_sheets">Google Sheets</option>
              <option value="s3">AWS S3</option>
              <option value="azure_blob">Azure Blob Storage</option>
              <option value="local">Local File System</option>
            </select>
          </div>

          {/* Service Account JSON File (Optional) */}
          <div>
            <label htmlFor="serviceAccount" className="block text-sm font-medium text-zinc-300 mb-1">
              Service Account Key (JSON File) <span className="text-zinc-500 text-xs">(Optional)</span>
            </label>
            <div className="space-y-2">
              <input
                ref={fileInputRef}
                type="file"
                id="serviceAccount"
                accept=".json,application/json"
                onChange={handleFileChange}
                className="w-full rounded-lg border border-white/10 bg-white/5 py-2 px-3 text-sm text-white file:mr-4 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-violet-500/20 file:text-violet-300 hover:file:bg-violet-500/30 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
              />
              {serviceAccountFile && (
                <p className="text-xs text-zinc-400">
                  Selected: {serviceAccountFile.name}
                </p>
              )}
              <p className="text-xs text-zinc-500">
                Upload a service account JSON file for authentication (optional)
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={handleClose}
              className="flex-1 rounded-lg border border-white/10 bg-white/5 py-2 px-4 text-sm font-medium text-white hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 rounded-lg bg-violet-500 py-2 px-4 text-sm font-medium text-white hover:bg-violet-600 transition-colors"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
