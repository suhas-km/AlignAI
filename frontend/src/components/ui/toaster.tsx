"use client"

import * as React from "react"
import { useTheme } from "next-themes"
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast"
import { useToast } from "@/components/ui/use-toast"

export function Toaster() {
  const { toasts } = useToast()
  const { theme } = useTheme()

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          <Toast
            key={id}
            className={`${theme === 'dark' ? 'bg-gray-900 border-gray-800' : 'bg-white border-gray-200'}`}
            {...props}
          >
            <div className="grid gap-1">
              {title && <ToastTitle className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>{title}</ToastTitle>}
              {description && (
                <ToastDescription className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                  {description}
                </ToastDescription>
              )}
            </div>
            {action}
            <ToastClose className={theme === 'dark' ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-900'} />
          </Toast>
        )
      })}
      <ToastViewport />
    </ToastProvider>
  )
}
