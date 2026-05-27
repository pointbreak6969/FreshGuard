'use client'

import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'
import {
  Upload,
  ImageIcon,
  Loader2,
  AlertCircle,
  Leaf,
  X,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ScanSearch,
} from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────────────

type Detection = {
  file: string
  label: string
  confidence: number
}

type AnalysisResult = {
  image_id: string
  results: Detection[]
}

type UploadState =
  | { status: 'idle' }
  | { status: 'dragging' }
  | { status: 'preview'; file: File; objectUrl: string }
  | { status: 'loading'; file: File; objectUrl: string }
  | { status: 'done'; file: File; objectUrl: string; data: AnalysisResult }
  | { status: 'error'; message: string }

// ── Label parsing helpers ──────────────────────────────────────────────────────

// Handles both CamelCase ("RottenPotato", "FreshBellpepper") and
// underscore-separated ("apple_fresh", "banana_rotten") labels.
function parseLabel(label: string): { fruit: string; condition: string } {
  const KNOWN_CONDITIONS = ['rotten', 'fresh', 'overripe', 'stale', 'spoiled']

  // Search for a known condition keyword anywhere in the label (case-insensitive)
  const lower = label.toLowerCase()
  for (const cond of KNOWN_CONDITIONS) {
    if (lower.includes(cond)) {
      // Strip the condition word (any case) to get the fruit name
      const fruitRaw = label.replace(new RegExp(cond, 'gi'), '')
      // Split residual CamelCase and clean up whitespace / underscores
      const fruit = fruitRaw
        .replace(/_/g, ' ')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/\s+/g, ' ')
        .trim()
      return { fruit: fruit || label, condition: cond }
    }
  }

  // Fallback: try underscore split (e.g. "apple_fresh")
  const parts = label.split('_')
  if (parts.length >= 2) {
    const condition = parts[parts.length - 1].toLowerCase()
    const fruit = parts
      .slice(0, -1)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')
    return { fruit, condition }
  }

  // No condition found — return the label as-is with empty condition
  return { fruit: label, condition: '' }
}

function getConditionMeta(condition: string): {
  label: string
  icon: React.ReactNode
  badgeClass: string
  barClass: string
} | null {
  const c = condition.toLowerCase()
  if (c === 'fresh') {
    return {
      label: 'Fresh',
      icon: <CheckCircle2 className="size-3.5" />,
      badgeClass:
        'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-400',
      barClass: '[&>div]:bg-emerald-500',
    }
  }
  if (c === 'rotten') {
    return {
      label: 'Rotten',
      icon: <XCircle className="size-3.5" />,
      badgeClass:
        'border-red-200 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-400',
      barClass: '[&>div]:bg-red-500',
    }
  }
  if (c.includes('over') || c === 'overripe') {
    return {
      label: 'Overripe',
      icon: <AlertTriangle className="size-3.5" />,
      badgeClass:
        'border-orange-200 bg-orange-50 text-orange-700 dark:border-orange-800 dark:bg-orange-950/40 dark:text-orange-400',
      barClass: '[&>div]:bg-orange-500',
    }
  }
  if (c === 'stale' || c === 'spoiled') {
    return {
      label: c.charAt(0).toUpperCase() + c.slice(1),
      icon: <AlertCircle className="size-3.5" />,
      badgeClass: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-400',
      barClass: '[&>div]:bg-amber-500',
    }
  }
  // Unknown / empty condition — return null so no badge is rendered
  return null
}

function getSummaryStats(results: Detection[]) {
  const parsed = results.map((r) => parseLabel(r.label))
  const fresh = parsed.filter((p) => p.condition === 'fresh').length
  const rotten = parsed.filter((p) => p.condition === 'rotten').length
  const other = results.length - fresh - rotten
  return { total: results.length, fresh, rotten, other }
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function DetectionCard({ detection, index }: { detection: Detection; index: number }) {
  const { fruit, condition } = parseLabel(detection.label)
  const meta = getConditionMeta(condition)
  const pct = Math.round(detection.confidence * 100)

  // Default bar colour when no condition is recognised
  const barClass = meta?.barClass ?? '[&>div]:bg-muted-foreground'

  return (
    <div className="flex items-start gap-3 rounded-lg border border-border bg-background p-3 transition-shadow hover:shadow-sm">
      {/* Index bubble */}
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">
        {index + 1}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-2">
        {/* Top row */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium capitalize">{fruit}</span>
          {meta && (
            <span
              className={cn(
                'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium',
                meta.badgeClass
              )}
            >
              {meta.icon}
              {meta.label}
            </span>
          )}
        </div>

        {/* Confidence bar */}
        <div className="flex items-center gap-2">
          <Progress
            value={pct}
            className={cn('h-1.5 flex-1', barClass)}
          />
          <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
            {pct}%
          </span>
        </div>
      </div>
    </div>
  )
}

function SummaryBadge({
  count,
  label,
  colorClass,
}: {
  count: number
  label: string
  colorClass: string
}) {
  return (
    <div className={cn('flex items-center gap-2 rounded-lg border px-3 py-2', colorClass)}>
      <span className="text-xl font-bold tabular-nums leading-none">{count}</span>
      <span className="text-xs font-medium leading-tight">{label}</span>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function FreshGuardUploader() {
  const [state, setState] = useState<UploadState>({ status: 'idle' })
  const inputRef = useRef<HTMLInputElement>(null)

  const acceptFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setState({ status: 'error', message: 'Please upload an image file (JPEG, PNG, WebP, etc.).' })
      return
    }
    // Revoke any old object URL
    if ('objectUrl' in state && state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl)
    }
    const objectUrl = URL.createObjectURL(file)
    setState({ status: 'preview', file, objectUrl })
  }, [state])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files?.[0]
      if (file) acceptFile(file)
    },
    [acceptFile]
  )

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) acceptFile(file)
    // Reset input so re-selecting the same file works
    e.target.value = ''
  }

  const clearFile = () => {
    if ('objectUrl' in state && state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl)
    }
    setState({ status: 'idle' })
  }

  const analyse = async () => {
    if (state.status !== 'preview') return
    const { file, objectUrl } = state

    setState({ status: 'loading', file, objectUrl })

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err?.error ?? `Server error ${res.status}`)
      }

      const data: AnalysisResult = await res.json()
      setState({ status: 'done', file, objectUrl, data })
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred.'
      setState({ status: 'error', message })
    }
  }

  const reset = () => {
    if ('objectUrl' in state && state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl)
    }
    setState({ status: 'idle' })
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const isLoading = state.status === 'loading'

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-4 py-10">
      {/* Header */}
      <div className="flex flex-col items-center gap-2 text-center">
        <div className="flex items-center gap-2">
          <div className="flex size-9 items-center justify-center rounded-xl bg-emerald-600 text-white shadow">
            <Leaf className="size-5" />
          </div>
          <span className="text-2xl font-semibold tracking-tight">FreshGuard</span>
        </div>
        <p className="text-sm text-muted-foreground">
          Upload a photo of your produce and get an instant freshness analysis.
        </p>
      </div>

      {/* Upload card */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Image</CardTitle>
          <CardDescription>
            Drag &amp; drop an image, or click to browse. Supports JPEG, PNG, WebP.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {/* Drop zone / preview */}
          {state.status === 'idle' || state.status === 'dragging' ? (
            <div
              className={cn(
                'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors select-none',
                state.status === 'dragging'
                  ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20'
                  : 'border-border bg-muted/30 hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/10'
              )}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setState({ status: 'dragging' }) }}
              onDragLeave={() => setState({ status: 'idle' })}
              onDrop={handleDrop}
            >
              <div className="flex size-12 items-center justify-center rounded-full bg-muted">
                <Upload className="size-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-medium">Drop your image here</p>
                <p className="mt-0.5 text-xs text-muted-foreground">or click to browse files</p>
              </div>
            </div>
          ) : (
            /* Image preview */
            <div className="relative overflow-hidden rounded-xl border border-border bg-muted/20">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={'objectUrl' in state ? state.objectUrl : ''}
                alt="Upload preview"
                className="max-h-72 w-full object-contain"
              />
              {/* Clear button */}
              {!isLoading && (
                <button
                  onClick={clearFile}
                  className="absolute right-2 top-2 flex size-7 items-center justify-center rounded-full bg-background/80 shadow backdrop-blur-sm transition-opacity hover:opacity-80"
                  aria-label="Remove image"
                >
                  <X className="size-4" />
                </button>
              )}
              {/* Loading overlay */}
              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-background/60 backdrop-blur-sm">
                  <Loader2 className="size-8 animate-spin text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-700">Analysing…</span>
                </div>
              )}
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleInputChange}
          />

          {/* Error */}
          {state.status === 'error' && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2.5 text-sm text-destructive">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <span>{state.message}</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2">
            {state.status === 'preview' && (
              <>
                <Button onClick={analyse} className="flex-1 bg-emerald-600 hover:bg-emerald-700">
                  <ScanSearch />
                  Analyse Freshness
                </Button>
                <Button variant="outline" onClick={clearFile}>
                  Clear
                </Button>
              </>
            )}
            {state.status === 'done' && (
              <Button variant="outline" onClick={reset} className="w-full">
                <Upload />
                Upload Another Image
              </Button>
            )}
            {state.status === 'error' && (
              <Button variant="outline" onClick={reset} className="w-full">
                Try Again
              </Button>
            )}
            {(state.status === 'idle' || state.status === 'dragging') && (
              <Button
                variant="outline"
                className="w-full"
                onClick={() => inputRef.current?.click()}
              >
                <ImageIcon />
                Choose Image
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {state.status === 'done' && (
        <div className="flex flex-col gap-4">
          {/* Summary */}
          <div>
            <h2 className="mb-3 text-base font-semibold">Analysis Results</h2>
            {state.data.results.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center gap-2 py-8 text-center text-muted-foreground">
                  <AlertCircle className="size-8 opacity-50" />
                  <p className="text-sm">No produce detected in this image.</p>
                  <p className="text-xs">Try a clearer photo with visible fruits or vegetables.</p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Stats row */}
                {(() => {
                  const stats = getSummaryStats(state.data.results)
                  return (
                    <div className="mb-4 grid grid-cols-3 gap-2">
                      <SummaryBadge
                        count={stats.total}
                        label="Items detected"
                        colorClass="border-border bg-muted/30 text-foreground"
                      />
                      <SummaryBadge
                        count={stats.fresh}
                        label="Fresh"
                        colorClass="border-emerald-200 bg-emerald-50/60 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-400"
                      />
                      <SummaryBadge
                        count={stats.rotten}
                        label="Not fresh"
                        colorClass="border-red-200 bg-red-50/60 text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400"
                      />
                    </div>
                  )
                })()}

                <Separator className="mb-4" />

                {/* Detection list */}
                <div className="flex flex-col gap-2">
                  {state.data.results.map((d, i) => (
                    <DetectionCard key={`${d.label}-${i}`} detection={d} index={i} />
                  ))}
                </div>

                <p className="mt-2 text-center text-xs text-muted-foreground">
                  Image ID: <span className="font-mono">{state.data.image_id}</span>
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
