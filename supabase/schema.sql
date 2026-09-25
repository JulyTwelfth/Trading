-- Run this file once in your own Supabase project's SQL Editor.
-- The backend connects with SUPABASE_SERVICE_KEY. RLS is enabled with no
-- browser-client policies, so anon/authenticated clients cannot read secrets.

create table if not exists public.licenses (
    key text primary key,
    active boolean not null default true,
    expires_at timestamptz,
    user_email text,
    created_at timestamptz not null default now()
);

create table if not exists public.sessions (
    key text primary key references public.licenses(key) on delete cascade,
    created_at timestamptz not null default now()
);

create table if not exists public.wallets (
    license_key text not null references public.licenses(key) on delete cascade,
    wallet_id text not null,
    proxy_address text not null,
    -- The existing application stores this value in plaintext. Use only a
    -- dedicated low-value bot signer; never store a main-wallet private key.
    private_key text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (license_key, wallet_id),
    constraint wallets_wallet_id_not_blank check (length(trim(wallet_id)) > 0),
    constraint wallets_proxy_address_format check (
        proxy_address ~* '^(0x)?[0-9a-f]{40}$'
    ),
    constraint wallets_private_key_format check (
        private_key ~* '^(0x)?[0-9a-f]{64}$'
    )
);

create table if not exists public.blacklisted_markets (
    license_key text not null references public.licenses(key) on delete cascade,
    condition_id text not null,
    slug text,
    question text,
    market_url text,
    created_at timestamptz not null default now(),
    primary key (license_key, condition_id)
);

create index if not exists blacklisted_markets_created_at_idx
    on public.blacklisted_markets (license_key, created_at);

alter table public.licenses enable row level security;
alter table public.sessions enable row level security;
alter table public.wallets enable row level security;
alter table public.blacklisted_markets enable row level security;

revoke all on table public.licenses from anon, authenticated;
revoke all on table public.sessions from anon, authenticated;
revoke all on table public.wallets from anon, authenticated;
revoke all on table public.blacklisted_markets from anon, authenticated;

grant all on table public.licenses to service_role;
grant all on table public.sessions to service_role;
grant all on table public.wallets to service_role;
grant all on table public.blacklisted_markets to service_role;
