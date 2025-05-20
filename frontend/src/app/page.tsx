"use client"

import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowRight, Code, Shield, Activity } from "lucide-react"
import { motion } from "framer-motion"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-gray-900 via-gray-800 to-black opacity-80"></div>
        
        <div className="container relative z-10 mx-auto px-4 py-24 sm:px-6 lg:flex lg:h-screen lg:items-center lg:px-8">
          <div className="max-w-3xl text-center sm:text-left">
            <motion.h1 
              className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl md:text-6xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              AI Alignment & Ethical Guardrails
            </motion.h1>
            
            <motion.p 
              className="mt-6 max-w-lg text-lg leading-8 text-gray-300 sm:text-xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              Ensure responsible, transparent, and safe AI interactions with our comprehensive alignment and compliance platform.
            </motion.p>
            
            <motion.div 
              className="mt-10"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <Button 
                asChild 
                size="lg" 
                className="bg-white text-black hover:bg-gray-200 px-8 py-6 text-lg"
              >
                <Link href="/analyzer">
                  Get Started with AlignAI
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-4xl">
              Features
            </h2>
            <p className="mx-auto mt-4 max-w-2xl text-xl text-gray-600 dark:text-gray-300">
              Powerful tools to ensure AI safety and compliance
            </p>
          </div>

          <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
            {[
              {
                icon: <Shield className="h-8 w-8 text-white" />,
                title: "Policy Compliance",
                description: "Ensure your AI models comply with ethical guidelines and regulations."
              },
              {
                icon: <Activity className="h-8 w-8 text-white" />,
                title: "Real-time Analysis",
                description: "Get instant feedback on potential issues in your AI outputs."
              },
              {
                icon: <Code className="h-8 w-8 text-white" />,
                title: "Developer Friendly",
                description: "Intuitive tools and a straightforward interface for seamless integration into your workflow."
              }
            ].map((feature, index) => (
              <motion.div 
                key={index}
                className="rounded-xl bg-white p-8 shadow-lg dark:bg-gray-800"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-black dark:bg-white/10 mb-6">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {feature.title}
                </h3>
                <p className="mt-2 text-gray-600 dark:text-gray-300">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gray-900 py-24">
        <div className="container mx-auto px-4 text-center sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Ready to get started?
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-xl text-gray-300">
            Start building safer AI applications with AlignAI's powerful analysis capabilities.
          </p>
          <div className="mt-8">
            <Button 
              asChild 
              size="lg" 
              className="bg-white text-black hover:bg-gray-200"
            >
              <Link href="/analyzer">
                Start Analyzing
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}
