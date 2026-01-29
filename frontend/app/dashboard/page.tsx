import { DashboardHeader } from "@/components/dashboard-header"
import { FileUploadSection } from "@/components/file-upload-section"
import { RecentAnalyses } from "@/components/recent-analyses"
import { StatsCards } from "@/components/stats-cards"
import { SystemStatus } from "@/components/system-status"
import { ApiTester } from "@/components/api-tester"
import { RealtimeStatus } from "@/components/realtime-status"
import { BatchPrediction } from "@/components/batch-prediction"

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />

      <main className="container mx-auto px-4 py-8 space-y-8">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold text-foreground">Research Dashboard</h1>
          <p className="text-muted-foreground">Upload methylation data and analyze Alzheimer's disease risk patterns</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-4">
          <SystemStatus />
          <RealtimeStatus />
        </div>

        <StatsCards />

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            <FileUploadSection />
            <BatchPrediction />
            <ApiTester />
          </div>
          <div>
            <RecentAnalyses />
          </div>
        </div>
      </main>
    </div>
  )
}
