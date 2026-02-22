import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;
  currentPage: string;
  selectedStudy: string;
  selectedSite: string;
  refreshKey: number;
  lastRefresh: Date | null;
  
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebarCollapsed: () => void;
  setCurrentPage: (page: string) => void;
  setSelectedStudy: (study: string) => void;
  setSelectedSite: (site: string) => void;
  triggerRefresh: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      sidebarOpen: false,
      sidebarCollapsed: false,
      currentPage: 'dashboard',
      selectedStudy: 'all',
      selectedSite: 'all',
      refreshKey: 0,
      lastRefresh: null,
      
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebarCollapsed: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setCurrentPage: (page) => set({ currentPage: page }),
      setSelectedStudy: (study) => set({ selectedStudy: study }),
      setSelectedSite: (site) => set({ selectedSite: site }),
      triggerRefresh: () => set((state) => ({ 
        refreshKey: state.refreshKey + 1, 
        lastRefresh: new Date() 
      })),
    }),
    {
      name: 'a6on-i-app-store',
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        selectedStudy: state.selectedStudy,
        selectedSite: state.selectedSite,
      }),
    }
  )
);
