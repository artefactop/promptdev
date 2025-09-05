#!/bin/bash
# Simple release script using uv's built-in version management

set -e

# Check if required tools are installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is required but not installed."
    echo "Install it with:"
    echo "  brew install gh"
    echo "  or visit: https://cli.github.com/"
    exit 1
fi

if ! command -v uvx &> /dev/null; then
    echo "❌ uvx is required but not installed."
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check if version type is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/release.sh <version_type>"
    echo "Version types:"
    echo "  patch  - 0.0.1 -> 0.0.2"
    echo "  minor  - 0.0.1 -> 0.1.0"
    echo "  major  - 0.0.1 -> 1.0.0"
    echo ""
    echo "Or specify exact version:"
    echo "  ./scripts/release.sh 0.0.2"
    exit 1
fi

VERSION_ARG=$1

# Check if it's a version type or exact version
if [[ "$VERSION_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Exact version provided
    echo "Setting exact version to $VERSION_ARG"
    
    # Show current version
    echo "Current version:"
    grep "version" pyproject.toml | head -1
    
    # Set exact version by updating pyproject.toml directly
    echo "Setting version to $VERSION_ARG..."
    # Use sed to replace the version line in pyproject.toml
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS requires -i '' for in-place editing
        sed -i '' "s/^version = \".*\"/version = \"$VERSION_ARG\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$VERSION_ARG\"/" pyproject.toml
    fi
    
    # Show new version
    NEW_VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)
    echo "New version: $NEW_VERSION"
else
    # Version type provided (patch, minor, major)
    echo "Incrementing $VERSION_ARG version..."
    
    # Map common terms to uv-version terms
    case $VERSION_ARG in
        "patch")
            UV_VERSION_TYPE="micro"
            ;;
        "minor")
            UV_VERSION_TYPE="minor"
            ;;
        "major")
            UV_VERSION_TYPE="major"
            ;;
        *)
            echo "❌ Unknown version type: $VERSION_ARG"
            echo "Use: patch, minor, or major"
            exit 1
            ;;
    esac
    
    # Show current version
    echo "Current version:"
    grep "version" pyproject.toml | head -1
    
    # Increment version using uv-version
    echo "Incrementing $UV_VERSION_TYPE version..."
    uvx uv-version increment $UV_VERSION_TYPE
    
    # Show new version
    NEW_VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)
    echo "New version: $NEW_VERSION"
fi

# Get the new version for git operations
NEW_VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)

# Get current version from git (last tag) for comparison
CURRENT_GIT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "No previous tags")

echo ""
echo "📋 Release Summary"
echo "=================="
echo "Current release: $CURRENT_GIT_VERSION"
echo "New version:     v$NEW_VERSION"
echo ""

# Confirmation prompt
read -p "🤔 Do you want to proceed with this release? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Release cancelled."
    # Restore the original version if we made changes
    git checkout -- pyproject.toml 2>/dev/null || true
    exit 1
fi

echo ""
echo "🚀 Creating release v$NEW_VERSION"
echo "======================="

# Git operations
echo "Creating release branch..."
RELEASE_BRANCH="release/v$NEW_VERSION"
git checkout -b $RELEASE_BRANCH

echo "Adding pyproject.toml..."
git add pyproject.toml

echo "Committing version bump..."
git commit -m "bump: version $NEW_VERSION"

echo "Pushing release branch..."
git push origin $RELEASE_BRANCH

echo "Creating pull request..."
gh pr create \
    --title "Release v$NEW_VERSION" \
    --body "🚀 Automated release for v$NEW_VERSION

## Changes
- Bump version to $NEW_VERSION
- Ready for release to PyPI

## Release Notes
This PR will trigger the release workflow once merged." \
    --base main \
    --head $RELEASE_BRANCH

echo "Enabling auto-merge (will merge when CI passes)..."
gh pr merge --auto --squash

echo "Creating GitHub release (will be created after PR merge)..."
echo "Note: GitHub release will be created automatically by the release workflow."

echo ""
echo "🎉 Release PR for v$NEW_VERSION created!"
echo "📋 The PR will auto-merge once CI passes."
echo "🚀 After merge, GitHub Actions will automatically publish to PyPI."
echo ""
echo "Monitor progress:"
echo "- PR: $(gh pr view --json url --jq .url)"
echo "- Actions: https://github.com/artefactop/promptdev/actions"
