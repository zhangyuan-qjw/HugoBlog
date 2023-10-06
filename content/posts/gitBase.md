---
title: "Git Base"
date: 2023-09-26
draft: false

tags: ["git"]
categories: ["English"]
---
# Basic commant of git
![An image](/img/git.png)
#### View currently owned baranches

>git branch

#### Basic project process

>git init
>
>git add .
>
>git commit -m "First commit"git
>
>git remote add origin <github-repo-url>
>
>git  push -u origin master

#### Update project process

>git status
>
>git add .
>
>git commit - "Description"
>
>git push (Default current branch)

#### when the romote project changes but the local project does not

>git pull origin master (Pull the latest changed remote branch)
>
>git add .
>
>git commit -m "merge local code"

#### what is the code branch in git used for? Generally I seem to only use master

>In Git, branch is an import concept used to manage and isolate different workflows.
>
>Branch allow you to do different works,fix bugs,add new features,etc.without affecting the main code line (usually the `master`branch).Here are some importmant concepts and  uses of branches: 
>
>1. **Master Branch**：The`master` branch is usually the defualt master branch of a Git repository .It contains stable,production-ready code.New features and fixes are usually deveploped on other branches and then intereated into `master` branch via merge or reset.
>
>2. **Feature Branches**：Feature branches are used to develop new features. When you want to add new features, you create a new branch and develop on that branch. Once feature development is complete, you can merge that branch back into the master branch (usually `master`).
>
>3. **Bugfix Branches**：Bugfix branches are used to resolve bugs. If you find a problem on the master branch, you can create a fix branch and fix the problem on that branch. Once the fix is complete, merge the fix branch back to the main branch.
>
>4. **Release Branches**：Release branches are used to prepare releases. When your software is about to be released, you can create a release branch for final testing and fixes. Once ready, the release branch can be merged into the master branch and the release version deployed to production.
>
>5. **Hotfix Branches**：Hotfix branches are used to quickly fix urgent problems in production environments, usually without going through the development and testing process. Once the fix is complete, the hotfix branch can be merged back into the master branch and other appropriate branches.
>