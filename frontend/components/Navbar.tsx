import Link from "next/link";
import FeedbackButton from "./FeedbackButton";

const Navbar = () => {
  return (
    <nav className="flex justify-between px-16 py-8 items-center">
      <h1 className="font-semibold text-3xl text-slate-200 font-logo z-10">
        datalens
      </h1>

      <ul className="flex gap-6 items-center justify-center font-semibold text-slate-200 font-logo z-10">
        <li>
          <FeedbackButton />
        </li>
        <li>
          <button className="inline-flex h-10 items-center justify-center rounded-full border border-slate-800 bg-[linear-gradient(110deg,#000103,45%,#1e2631,55%,#000103)] bg-[length:200%_100%] px-6 font-medium text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50">
            <Link href="/playground">Playground</Link>
          </button>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
