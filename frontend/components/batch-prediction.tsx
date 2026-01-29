"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { api } from "@/lib/axios";

interface BatchFile {
  id: string;
  file: File;
  status: "pending" | "processing" | "completed" | "error";
  result?: any;
  error?: string;
}

export function BatchPrediction() {
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        status: "pending" as const,
      }));
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const processBatch = async () => {
    setProcessing(true);
    setProgress(0);

    const total = files.length;
    let completed = 0;

    for (const file of files) {
      if (file.status !== "pending") continue;

      try {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id ? { ...f, status: "processing" } : f
          )
        );

        const formData = new FormData();
        formData.append("file", file.file);
        formData.append("model_type", "both");

        const response = await api.predictions.predict(formData);

        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? { ...f, status: "completed", result: response }
              : f
          )
        );
      } catch (error: any) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? { ...f, status: "error", error: error.message }
              : f
          )
        );
      }

      completed++;
      setProgress((completed / total) * 100);
    }

    setProcessing(false);
  };

  const clearFiles = () => {
    setFiles([]);
    setProgress(0);
  };

  const getStatusIcon = (status: BatchFile["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: BatchFile["status"]) => {
    const variants = {
      pending: "secondary",
      processing: "default",
      completed: "outline",
      error: "destructive",
    } as const;

    return (
      <Badge variant={variants[status]}>
        {status}
      </Badge>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Batch Predictions
        </CardTitle>
        <CardDescription>
          Upload multiple files for batch processing
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <label htmlFor="batch-upload" className="cursor-pointer">
            <Button variant="outline" asChild>
              <span>
                <Upload className="mr-2 h-4 w-4" />
                Select Files
              </span>
            </Button>
            <input
              id="batch-upload"
              type="file"
              multiple
              accept=".csv,.txt"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>

          <Button
            onClick={processBatch}
            disabled={files.length === 0 || processing}
          >
            {processing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Start Batch"
            )}
          </Button>

          <Button
            variant="outline"
            onClick={clearFiles}
            disabled={processing}
          >
            Clear All
          </Button>
        </div>

        {files.length > 0 && (
          <>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-medium">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} />
            </div>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-3 border rounded-lg"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {getStatusIcon(file.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {file.file.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {(file.file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(file.status)}
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-4 gap-2 pt-2 border-t">
              <div className="text-center">
                <p className="text-2xl font-bold">{files.length}</p>
                <p className="text-xs text-muted-foreground">Total</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">
                  {files.filter((f) => f.status === "processing").length}
                </p>
                <p className="text-xs text-muted-foreground">Processing</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">
                  {files.filter((f) => f.status === "completed").length}
                </p>
                <p className="text-xs text-muted-foreground">Completed</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600">
                  {files.filter((f) => f.status === "error").length}
                </p>
                <p className="text-xs text-muted-foreground">Errors</p>
              </div>
            </div>
          </>
        )}

        {files.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            No files selected. Click "Select Files" to add files for batch processing.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
