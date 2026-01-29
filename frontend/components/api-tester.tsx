"use client"

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Play, Check, X, Loader2 } from 'lucide-react';
import { api } from '@/lib/axios';

interface TestResult {
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  message?: string;
  duration?: number;
}

export function ApiTester() {
  const [results, setResults] = useState<TestResult[]>([]);
  const [testing, setTesting] = useState(false);

  const tests = [
    {
      name: 'Health Check',
      test: async () => {
        const response = await api.health.check();
        return `API is ${response.data.status}`;
      },
    },
    {
      name: 'System Info',
      test: async () => {
        const response = await api.health.systemInfo();
        return `Python ${response.data.python_version.split(' ')[0]}`;
      },
    },
    {
      name: 'Models Info',
      test: async () => {
        const response = await api.models.info();
        return `${response.data.total_models} models loaded`;
      },
    },
    {
      name: 'XGBoost Status',
      test: async () => {
        const response = await api.models.status('xgboost');
        return `Model is ${response.data.status}`;
      },
    },
    {
      name: 'PyTorch Status',
      test: async () => {
        const response = await api.models.status('pytorch');
        return `Model is ${response.data.status}`;
      },
    },
  ];

  const runTests = async () => {
    setTesting(true);
    setResults([]);

    for (const { name, test } of tests) {
      setResults(prev => [...prev, { name, status: 'running' }]);
      const startTime = Date.now();

      try {
        const message = await test();
        const duration = Date.now() - startTime;
        
        setResults(prev =>
          prev.map(r =>
            r.name === name
              ? { ...r, status: 'success', message, duration }
              : r
          )
        );
      } catch (error: any) {
        const duration = Date.now() - startTime;
        const message = error?.response?.data?.message || error?.message || 'Test failed';
        
        setResults(prev =>
          prev.map(r =>
            r.name === name
              ? { ...r, status: 'error', message, duration }
              : r
          )
        );
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setTesting(false);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case 'success':
        return <Check className="w-4 h-4 text-green-500" />;
      case 'error':
        return <X className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>API Connection Tester</CardTitle>
        <CardDescription>
          Test the connection between frontend and backend API
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button
          onClick={runTests}
          disabled={testing}
          className="w-full"
        >
          {testing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Testing...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Run Tests
            </>
          )}
        </Button>

        {results.length > 0 && (
          <div className="space-y-2">
            {results.map((result, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 border rounded-lg"
              >
                <div className="flex items-center gap-3 flex-1">
                  {getStatusIcon(result.status)}
                  <div className="flex-1">
                    <p className="font-medium text-sm">{result.name}</p>
                    {result.message && (
                      <p className="text-xs text-muted-foreground">{result.message}</p>
                    )}
                  </div>
                </div>
                
                {result.duration !== undefined && (
                  <Badge variant="secondary" className="text-xs">
                    {result.duration}ms
                  </Badge>
                )}
              </div>
            ))}
          </div>
        )}

        {results.length > 0 && !testing && (
          <Alert>
            <AlertDescription>
              {results.filter(r => r.status === 'success').length} of {results.length} tests passed
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}