import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
export const supabaseConfigured = Boolean(supabaseUrl && supabaseKey);

export const supabase = createClient(
  supabaseUrl || "https://example.supabase.co",
  supabaseKey || "local-dev-anon-key"
);
