import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    timeout: 120000,
    headers: {
        "Content-Type": "application/json",
    },
});

axiosInstance.interceptors.request.use(
    (config) => {
        console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
    },
    (error) => {
        console.error("[API Request Error]", error);
        return Promise.reject(error);
    }
);

axiosInstance.interceptors.response.use(
    (response: AxiosResponse) => {
        console.log(`[API Response] ${response.status} ${response.config.url}`);
        return response;
    },
    (error: AxiosError) => {
        console.error("[API Response Error]", {
            status: error.response?.status,
            message: error.message,
            data: error.response?.data,
        });

        if (error.code === "ECONNABORTED") {
            console.error("Request timeout");
        } else if (!error.response) {
            console.error("Network error - server might be down");
        }

        return Promise.reject(error);
    }
);

export const api = {
    health: {
        check: () => axiosInstance.get("/health"),
        systemInfo: () => axiosInstance.get("/system/info"),
    },
    
    models: {
        info: () => axiosInstance.get("/models/info"),
        status: (modelType: string) => axiosInstance.get(`/models/${modelType}/status`),
    },
    
    predictions: {
        predict: (formData: FormData) => 
            axiosInstance.post("/predict", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            }),
    },
    
    analyses: {
        list: () => axiosInstance.get("/analyses"),
        get: (id: string) => axiosInstance.get(`/analyses/${id}`),
        delete: (id: string) => axiosInstance.delete(`/analyses/${id}`),
    },
};