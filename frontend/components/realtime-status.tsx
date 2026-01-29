"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useStatusWebSocket } from "@/hooks/use-websocket";
import { Activity, Wifi, WifiOff } from "lucide-react";

export function RealtimeStatus() {
  const { isConnected, status } = useStatusWebSocket();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Real-time Status
            </CardTitle>
            <CardDescription>Live backend connection</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {isConnected ? (
              <>
                <Wifi className="h-4 w-4 text-green-500" />
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  Connected
                </Badge>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-500" />
                <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                  Disconnected
                </Badge>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {status ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Backend Status</span>
              <Badge variant={status.backend_status === "online" ? "default" : "destructive"}>
                {status.backend_status}
              </Badge>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-medium">Model Status</span>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex flex-col gap-1 p-2 border rounded-md">
                  <span className="text-xs text-muted-foreground">XGBoost</span>
                  <Badge
                    variant={status.models?.xgboost?.status === "ready" ? "default" : "secondary"}
                    className="w-fit"
                  >
                    {status.models?.xgboost?.status || "unknown"}
                  </Badge>
                </div>
                <div className="flex flex-col gap-1 p-2 border rounded-md">
                  <span className="text-xs text-muted-foreground">PyTorch</span>
                  <Badge
                    variant={status.models?.pytorch?.status === "ready" ? "default" : "secondary"}
                    className="w-fit"
                  >
                    {status.models?.pytorch?.status || "unknown"}
                  </Badge>
                </div>
              </div>
            </div>

            <div className="text-xs text-muted-foreground">
              Last update: {new Date().toLocaleTimeString()}
            </div>
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">
            Waiting for status updates...
          </div>
        )}
      </CardContent>
    </Card>
  );
}
