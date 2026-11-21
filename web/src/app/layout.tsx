import type { Metadata } from "next"
import { Noto_Sans_Arabic } from "next/font/google"
import "./globals.css"

const notoArabic = Noto_Sans_Arabic({ variable: "--font-noto-arabic", subsets: ["arabic"], weight: ["400", "500", "600"] })

export const metadata: Metadata = {
  title: "Hawal: a fine-tuned Kurdish news topic classifier",
  description: "KuBERT fine-tuned on 41,838 Central Kurdish (Sorani) news headlines. Runs in your browser.",
}

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={notoArabic.variable}>
      <body className="min-h-dvh font-sans">{children}</body>
    </html>
  )
}
