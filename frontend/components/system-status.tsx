"use client"

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, Server, Database, AlertCircle } from 'lucide-react';
import { useHealthCheck, useModelsInfo } from '@/hooks/use-api';

export function SystemStatus() {
  const { data: healthData, loading: healthLoading, execute: checkHealth } = useHealthCheck();
  const { data: modelsData, loading: modelsLoading, execute: fetchModels } = useModelsInfo();
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        await checkHealth();
        await fetchModels();
        setIsOnline(true);
      } catch (error) {
        setIsOnline(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: boolean) => {
    return status ? (
      <Badge variant="default" className="bg-green-500">
        <Activity className="w-3 h-3 mr-1" />
        Online
      </Badge>
    ) : (
      <Badge variant="destructive">
        <AlertCircle className="w-3 h-3 mr-1" />
        Offline
      </Badge>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Server className="w-5 h-5" />
            System Status
          </span>
          {getStatusBadge(isOnline)}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-sm font-medium flex items-center gap-2">
              <Database className="w-4 h-4" />
              Backend API
            </p>
            {healthLoading ? (
              <Badge variant="secondary">Checking...</Badge>
            ) : healthData ? (
              <Badge variant="default" className="bg-green-500">
                {healthData.status}
              </Badge>
            ) : (
              <Badge variant="destructive">Unavailable</Badge>
            )}
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium">API Version</p>
            <p className="text-sm text-muted-foreground">
              {healthData?.version || 'N/A'}
            </p>
          </div>
        </div>

        {modelsData && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Loaded Models</p>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(modelsData.models || {}).map(([key, value]: [string, any]) => (
                <Badge
                  key={key}
                  variant={value?.loaded ? "default" : "secondary"}
                  className={value?.loaded ? "bg-green-500" : ""}
                >
                  {key}: {value?.loaded ? 'Ready' : 'Not Loaded'}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {healthData?.timestamp && (
          <p className="text-xs text-muted-foreground">
            Last checked: {new Date(healthData.timestamp).toLocaleTimeString()}
          </p>
        )}
      </CardContent>
    </Card>
  );
}