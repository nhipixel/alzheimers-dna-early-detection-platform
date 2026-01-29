import { useState, useCallback } from 'react';
import { api } from '@/lib/axios';

interface UseApiOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
}

export function useApi<T = any>(apiCall: (...args: any[]) => Promise<any>, options?: UseApiOptions) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async (...args: any[]) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiCall(...args);
      const result = response.data;
      setData(result);
      
      if (options?.onSuccess) {
        options.onSuccess(result);
      }
      
      return result;
    } catch (err: any) {
      const errorMessage = err?.response?.data?.message || err?.message || 'An error occurred';
      setError(errorMessage);
      
      if (options?.onError) {
        options.onError(err);
      }
      
      throw err;
    } finally {
      setLoading(false);
    }
  }, [apiCall, options]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

export function useHealthCheck() {
  return useApi(api.health.check);
}

export function useModelsInfo() {
  return useApi(api.models.info);
}

export function usePrediction() {
  return useApi(api.predictions.predict);
}

export function useAnalysesList() {
  return useApi(api.analyses.list);
}