# Git and GitHub History Cleanup Guide

This note documents the cleanup we performed after generated files under
`artifacts/generated/` were accidentally committed and pushed to GitHub.

It is written as a Git learning guide, not just as a command log.

## What Happened

The push failed with:

```text
! refs/heads/main:refs/heads/main [rejected] (fetch first)
```

That meant the remote branch had commits that the local branch did not have.
After fetching, the repository was in this state:

```text
main...origin/main [ahead 1, behind 1]
```

Local `main` had one commit that was not on GitHub:

```text
e574621 Improve captioning workflow and application behavior
```

Remote `origin/main` had one commit that was not local:

```text
754142d Delete artifacts/generated directory
```

The remote commit deleted generated output files, but the local commit added
new files under the same generated-output path. Git rejected the push because
directly pushing local `main` would have overwritten the remote branch history.

## Key Git Concepts

### Commit

A commit is a snapshot of the tracked files, plus metadata and a pointer to its
parent commit.

Deleting a file in a later commit only removes it from later snapshots. It does
not erase the file from older commits.

### Branch

A branch is just a movable name pointing to one commit.

For example:

```text
main -> a52e7e5
```

When a new commit is made, `main` moves forward to point to that new commit.

### Remote-Tracking Branch

`origin/main` is your local record of where GitHub's `main` branch pointed at
the last fetch.

It is not updated automatically every second. You update it with:

```bash
git fetch origin
```

### Fast-Forward Push

A normal push is allowed when GitHub can move its branch pointer forward without
discarding commits.

This is allowed:

```text
A -- B -- C
          \
           D
```

GitHub can move from `C` to `D`.

This is rejected:

```text
A -- B -- C   origin/main
      \
       D      local main
```

Local and remote both have different commits after `B`. Git will not silently
discard `C`.

## Why `.gitignore` Was Needed

The path `artifacts/generated/` contains generated outputs. These files should
exist locally, but should not be versioned.

We added:

```gitignore
artifacts/generated/
```

Important rule:

`.gitignore` only prevents new untracked files from being added. It does not
remove files already tracked by Git.

For already tracked files, use:

```bash
git rm -r --cached artifacts/generated
```

`--cached` removes files from Git's index while leaving the local files on disk.

## First Cleanup: Fix the Latest Commit

The first cleanup made the latest branch state correct.

We reset the local commit back into the index, removed generated files from the
index, and recommitted only source/docs changes:

```bash
git reset --soft 2bbfc16
git rm -r --cached --ignore-unmatch artifacts/generated
git reset --mixed origin/main
git add .
git commit -m "Improve captioning workflow and application behavior"
git push origin main
```

After this, the current `main` tree no longer tracked generated files:

```bash
git ls-tree -r --name-only HEAD -- artifacts/generated | wc -l
```

Expected output:

```text
0
```

However, this did not clean old Git history.

## Why Deleting a File Is Not Enough

Suppose a file appears in commit `B` and is deleted in commit `C`:

```text
A -- B -- C
```

The latest commit `C` does not contain the file. But commit `B` still does.

Anyone who can access the repository history can inspect `B`:

```bash
git show B:path/to/file
```

So if the file must be removed from history, normal deletion is not enough. The
history itself must be rewritten.

## Full Cleanup: Rewrite History

To remove `artifacts/generated/` from all commits, we used `git-filter-repo`.

Install it:

```bash
brew install git-filter-repo
```

Create a local backup before rewriting:

```bash
git branch backup/pre-filter-artifacts-generated-20260805 HEAD
```

Rewrite history:

```bash
git filter-repo --path artifacts/generated/ --invert-paths --force
```

Meaning:

- `--path artifacts/generated/` selects that path.
- `--invert-paths` removes the selected path instead of keeping it.
- `--force` allows the rewrite in this existing repository.

`git-filter-repo` removed the `origin` remote as a safety measure, so we added
it back:

```bash
git remote add origin git@github.com:vickers-rz/whisper-captioner.git
```

Then we force-pushed the rewritten branch:

```bash
git push --force-with-lease origin main
```

Prefer `--force-with-lease` over `--force`. It refuses to overwrite the remote
if someone else pushed new work after your last fetch.

## Verification Commands

Check the remote refs:

```bash
git ls-remote --heads --tags origin
git ls-remote origin 'refs/pull/*'
```

In this case, GitHub had:

```text
refs/heads/main
```

No tags and no visible pull-request refs.

Check the current remote branch tree:

```bash
git ls-tree -r --name-only origin/main -- artifacts/generated | wc -l
```

Expected:

```text
0
```

Check whether the path appears anywhere in the remote branch history:

```bash
git log origin/main --oneline -- artifacts/generated
```

Expected: no output.

Check whether any object reachable from `origin/main` has that path:

```bash
git rev-list --objects origin/main -- artifacts/generated
```

Expected: no output.

Check that ignore works:

```bash
git check-ignore -v artifacts/generated/testfile
```

Expected output should point at `.gitignore`:

```text
.gitignore:14:artifacts/generated/ artifacts/generated/testfile
```

## What This Guarantees

After the rewrite and force push, anyone cloning the current GitHub repository
through normal Git refs cannot get `artifacts/generated/` from `main` history.

Normal commands such as these should not reveal the path:

```bash
git clone git@github.com:vickers-rz/whisper-captioner.git
git log --all -- artifacts/generated
git rev-list --objects --all -- artifacts/generated
```

## What This Cannot Fully Guarantee

History rewrite cannot control copies outside the rewritten repository.

Possible remaining copies:

- Someone cloned or fetched the repository before cleanup.
- A fork was created before cleanup.
- GitHub has temporary internal caches or old commit page caches.
- Logs, CI artifacts, or local backups stored the generated files elsewhere.

If the removed content is sensitive enough that cached copies matter, contact
GitHub Support and ask them to purge cached views and unreachable objects
related to the rewritten sensitive history.

## What Other Developers Need To Do

Because `main` history was rewritten, other local clones must resync.

If they have no local work to keep:

```bash
git fetch origin
git reset --hard origin/main
git gc --prune=now
```

If they have local work, they should save it first:

```bash
git switch -c my-work-backup
git fetch origin
git switch main
git reset --hard origin/main
```

Then they can reapply their work carefully by cherry-picking or rebasing.

## Practical Rules

- Put generated output directories in `.gitignore` before running scripts that
  create many files.
- Use `git status --short` before every commit.
- Use `git diff --cached --name-status` before committing.
- Use `git fetch origin` before pushing if there is any chance another machine
  or GitHub changed the branch.
- Do not use `git push --force` casually.
- Use `git push --force-with-lease` when a history rewrite is intentional.
- Deleting a file in Git removes it from the latest version, not from history.
- To remove a path from all history, use `git-filter-repo` or BFG.

