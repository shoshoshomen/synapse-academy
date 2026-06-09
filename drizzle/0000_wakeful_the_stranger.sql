CREATE TABLE "lesson_progress" (
	"user_email" text NOT NULL,
	"lesson_key" text NOT NULL,
	"completed" boolean DEFAULT false NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "lesson_progress_user_email_lesson_key_pk" PRIMARY KEY("user_email","lesson_key")
);
--> statement-breakpoint
CREATE TABLE "quiz_results" (
	"user_email" text NOT NULL,
	"lesson_key" text NOT NULL,
	"correct" integer NOT NULL,
	"total" integer NOT NULL,
	"passed" boolean NOT NULL,
	"at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "quiz_results_user_email_lesson_key_pk" PRIMARY KEY("user_email","lesson_key")
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text,
	"email" text NOT NULL,
	"password_hash" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
