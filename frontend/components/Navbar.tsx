import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import Link from "next/link";
import FeedbackButton from "./FeedbackButton";
import ThemeToggle from "./layout/ThemeToggle/theme-toggle";
import { Button } from "./ui/button";

const Navbar = () => {
  return (
    <nav className="border-b-[1px] sticky top-0 z-20 bg-clip-padding backdrop-filter backdrop-blur-3xl bg-opacity-50">
      <div className="flex items-center justify-between px-6 py-4 sm:px-28">
        <h1 className="font-semibold text-xl sm:text-3xl text-slate-800 dark:text-slate-200 font-logo z-10">
          datalens
        </h1>

        <div className="flex items-center justify-center gap-6">
          {/* Small Screen */}
          <div className="flex gap-3 sm:hidden">
            <DropdownMenu>
              <DropdownMenuTrigger>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="lucide lucide-align-right"
                >
                  <line x1="21" x2="3" y1="6" y2="6" />
                  <line x1="21" x2="9" y1="12" y2="12" />
                  <line x1="21" x2="7" y1="18" y2="18" />
                </svg>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>
                  <Link href="/playground">Playground</Link>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>
                  <a
                    href="https://github.com/Suhrud1511/DataLens"
                    target="_blank"
                  >
                    GitHub
                  </a>
                </DropdownMenuLabel>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Large Screen */}
          <div className="hidden sm:flex">
            <ul className="flex gap-4 items-center justify-center font-semibold text-slate-800 dark:text-slate-200 font-logo z-10">
              <li>
                <FeedbackButton />
              </li>
              <li>
                <button className="inline-flex h-10 items-center justify-center rounded-full border border-slate-800 bg-[linear-gradient(110deg,#000103,45%,#1e2631,55%,#000103)] bg-[length:200%_100%] px-6 font-medium text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50">
                  <Link href="/playground">Playground</Link>
                </button>
              </li>
              <li>
                <Button variant="outline" size="icon">
                  <a
                    href="https://github.com/Suhrud1511/DataLens"
                    target="_blank"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="lucide lucide-github"
                    >
                      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
                      <path d="M9 18c-4.51 2-5-2-7-2" />
                    </svg>
                  </a>
                </Button>
              </li>
            </ul>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
