import { setRequestLocale } from "next-intl/server";
import { getPreviewLessons, lessonKey } from "@/lib/content";
import {
  DashboardClient,
  type DashboardLesson,
} from "@/components/dashboard-client";

export default async function DashboardPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);

  const previews = await getPreviewLessons(locale);
  const lessons: DashboardLesson[] = previews.map(({ phase, lesson }) => ({
    key: lessonKey(phase.slug, lesson.slug),
    href: `/phases/${phase.slug}/${lesson.slug}`,
    phaseNumber: phase.phaseNumber,
    phaseTitle: phase.title,
    lessonTitle: lesson.title,
  }));

  return <DashboardClient lessons={lessons} />;
}
