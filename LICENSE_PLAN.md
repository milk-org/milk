# LICENSING NOTES AND ADDITIONAL CLAUSES

All code distributed within this repository for which the maintainers are the copyright holders,
are distributed under the Lesser GNU General Public Licence (LGPL), version 3 or subsequent.

> [!NOTE]
> This explanation may change, does not supersede legal requirements of all licensing and copyrights involved, and is delivered with no warranty whatsoever, including any warranty of being legally correct and accurate.
> If you find yourself in a situation where distributing a derivative work of MILK/CACAO under closed source, for serious money, or with significant legal requirements, we strongly suggest you have lawyers go over the entire licensing chain, starting at this repository and the dependencies thereof.
> If such a claim has any legal stance: we _require_ that you contact the maintainers and share the relevant findings of your legal team in regards to the maintainer's version of MILK/CACAO.

The ultimate license that applies to MILK/CACAO depends on your usage and intention thereof, and the repository contents are subject to:
- License escalation to the complete GNU General Public License (GPL)
- License deadlock due to the combination of a distributor with closed-source intentions, mixing in GPL cross-dependencies, and non-openable / non-GPL compatible cross-dependencies.

In this brief explanation, we consider 3 profiles:
- The MILK/CACAO maintainers team (MT)
- A 3rd party user person or organization (3PU)
- A 3rd party distributor person or organization (3PD)


## Third-party non-distributing users (3PUs)

This is the simplest role. Our understanding is that there are essentially no restrictions to your software usage  and modifications except for those caused by:
- Applicable law
- Proprietary licenses that end up combined with the MILK/CACAO software (practical example: API/SDKs of acquisition devices used as an RTC input).

You remain in this category as long as you do not distribute software or derived work and your usage remains private or within a single organization.
Distributing begins as soon as hosting your modified MILK/CACAO copy, or the derived work that results from combination with MILK/CACAO on a public facing repository (or private but beyond organization boundaries).

If these distributing conditions above apply, your position necessarily escalates to being a 3rd party distributor.

## Third-party distributors (3PDs) and Maintainer Team (MT)

### Default licensing for the distribution by the MT

The MT-distrib is the distributed software by the maintainer teams (ie organizations defined per `github.com/cacao-org` and `github.com/milk-org`). While separately hosted, this license scheme applies to CACAO.

A 3PD-distrib of MILK/CACAO is any expansion thereof by fork, clone, distribution of binaries, whether of the modified MT-CACAO, or optional modules/plugins/features/tooling that binds into this system and has no real utility without the original or modified contents of the MT-distrib.

In the default case (ie not the _GPL escalation_ case, see below for that):
- ImageStreamIO is distributed under MIT license (but be aware that it optionally depends on CUDA).
- MILK engine and framework is distributed under LGPL license.
- MILK modules (core, extras, cacao) from the MT-distrib:
  - that do not cause a license escalation are distributed under LGPL.
  - that combine with GPL dependencies are distributed under GPL.
  - that combine with proprietary dependencies are distributed in compliance with LGPL and Proprietary.
- MILK modules that may combine both GPL and proprietary GPL-incompatible dependencies (that may not be separated) __are not to be distributed by the MT nor any 3PD__.

MILK largely benefits from its plugin-designed architecture and seeks to leverage it to enable as many free usages as possible.
It is the MT's interpretration avoid that the plugin structure avoids licence creep from the modules into the core/engine/framework: it is the modules that are _derivatives_ of the core, and have the core as a dependency among their other dependencies.

Let us consider distribution interests that may apply to a 3PD-distrib:
1) Distribution of a modified MT distrib: __allowed__, under conservation of LGPL/GPL licensing scheme.
2) Distribution of modules under very permissive license (e.g. BSD or MIT): __allowed__.
3) Distribution of modules under GPL license: __allowed__.
4) Distribution of modules under proprietary closed source licenses: __allowed except for GPL escalation__.

### Key clauses / explanations

#### GPL escalation

Some ways of usage / compilation flags that are offered within the MT-distrib automatically induce a license escalation into the full GNU GPL:
- Enabling a GPL dependency, such a `gsl`, `readline`, etc.
- Static linking through LGPL code. This may be implied by using the LTO optimization option (todo: verify) or at least the GPL may consider so.

Falling under any of those cases, the MT-distrib (at least, all the portions affected, which may exclude some modules) __is considered distributed under GPL license__.

#### License deadlocking with GPL

Let's consider a library P that is distributed as binary + header only under a strict proprietary license.

A distributor develops a MILK module that mixes a dependency to a GPL library G and to library P. The exact details are beyond the writer's expertise, but we read the example as such:
- P is closed source. We assume it's not a very mainstream component that benefits from the system library exception.
- But because of the dependency on G, the module must be release under LGPL.
- Using P from the modules's GPL code is assumed is forbidden:
  - we do not concern ourselves with it being or not a P license infringement;
  - but it is a GPL infringement: it would require P to be open-source and under a GPL-compatible license.

Therefore, if the version of MILK/CACAO used (or the module wherein P is used) enables/requires GPL-escalating features to function, this results in impossible licensing requirements.

Therefore, __we must forbid__ the combined use of GPL and Proprietary features in MILK/CACAO, _at least at the distribution stage_. This requirement carries from the MT-distrib and any subsequent 3PD-distrib.
We propose to achieve this by implementing backstops in compilation options;
- MT or 3PD cannot distribute such that a combination of compilation/installation/usage options would lead to a license deadlock.
- MT or 3PD cannot distribute a script that "patches" the build toolchain automatically to circumvent this backstop.
- MT or 3PD cannot distribute instructions or document on how to circumvent this backstop, however trivial it may be.

For example: we could put a LICENSE_LOCK=ON in the CMakeLists.txt, that block compilation of incompatible modules. We can't tell people its there, but if a 3PU org were to acquire MILK, and voluntarily patch the relevant compilation backstop, and use the combined work:
- They would be permitted to do so, as final users.
- They would NOT be permitted to distribute.

> [!TBC] Code using CUDA benefits from the GPL's system library exception, and as such may not be part of the libraries that would cause license deadlock.

#### Why is the MT-distrib licensing allowable

- Dynamic linking to and from LGPL is allowed with only fairly weak copyleft requirements. The licensing is strong open-source but only for the library code, not the entire software work.
- We argue that modules (even those packaged as part of the MT-distrib) link only to the MILK engine, not across each other.
- We argue that modules that may have incompatible licenses only interact at IPC / OS level and that we distribute aggregated work, which usage is left to the user.

#### LGPL giveback

If a 3PD distributes a modified version of the LGPL portion of the MT-distrib, they are __required to open-source these modifications and publish these under LGPL or a superior license (e.g. GPL)__.

If published under LGPL, the 3PD-distrib may have its own GPL escalation derived from the one for the MT-distrib (e.g. additional modules).

Additionally, we request that the MT be notified of the release of derived work that performs modifications falling under the LGPL giveback, typically by means of a GitHub issue used as a discussion / messaging channel, or a GitHub pull request if the 3PD wishes to contribute code back to the MT-distrib.

#### More details on 3PD-distrib compliance requirements

We assume the case of a 3PD that is distributing a modified or expanded version of MILK/CACAO.
- _Restrictions_:
  - The same restrictions for the licensing of various modules as listed above apply.
  - Any modifications to LGPL files/features/modules shall be made open-source and distributed under either GPL or LGPL
  - See additional clauses

- _Permissivity_:
  - A 3PD may distribute any additional features and modules under only dynamic linking to the rest of MILK / the MILK engine, which does not cause any licensing requirement.
  - The 3PD's extension licensing may therefore be chosen and comply only with the internal requirements of the extension.
  - This allows most distribution cases, from fully open under MIT, hard copyleft under GPL, to fully closed with distribution of binaries only (except for the restrictions hereabove).

__However__, the 3PD's work/product may not be allowable to be distributed as a single work, depending on the licenses combined and the usage of MILK made/offered/distributed.

#### Copyright citation and acknowledgement

We request that usage of this work be cited and acknowledged.
- In derived software work (milk modules, expansion features of the MT-distrib that is part of the common work, modifications falling under the LGPL giveback) that does not already include this file and the `doc/acknowledgement.md` file.
- In presentations, by simple means of "Work using the MILK/CACAO RTC toolkit".
- In conference proceedings, technical reports, articles, and all forms of publications:
  - with the following acknowledgement:
todo: add acknowledgement.
  - or the following citation:
todo: add bibtex citation.

This citation/acknowledgement is released for publication discussing only scientific observations and advancements from data obtained with instruments operated with MILK/CACAO, as these works stand more remote from instrumental development realities.

## Simple Q&A

_Apply answers to any and all statements that are true._

1) I cloned repositories locally, only for personal or organizational use.
   - No particulars.
2) I cloned MILK/CACAO, and this clone is shared outside of my organization
   - Requirements for MT and 3PD apply.
   - GPL escalation reserve applies
   - License deadlocking applies
   - Copyright citation and acknowledgement clause applies
   - LGPL giveback clause applies
3) I forked MILK/CACAO, and this fork is publicly visible.
   - Same as 2. above.
4) I publish/distribute/publicly host an additional/optional MILK module.
   - License deadlocking applies
   - GPL escalation applies
   - Copyright citation and acknowledgement clause applies
5) My module requires a special feature / modification / proposed update to MILK/CACAO besides code autonomously packaged in my module
   - All of 4. above
   - LGPL giveback clause applies

_An example of requirements in case of strong opaque / strong copyright distribution_

You / your org. wishes to distribute MILK/CACAO and a series of novel, proprietary licensed modules, under closed source (e.g. precompiled binaries + headers only):
- You should ensure your modules are compliant with the licenses of their dependencies.
- Your license should carry the copyright citation and acknowledgement clause
- Your advertising, documentation, publications should comply with the citation and acknowledgement clause
- You cannot enable any GPL-escalation features of MILK, nor should you provide capability to your distributees to reallow them.

Under those conditions, your modules may be distributed commercially and opaquely if you so wish. If you require any modification to MILK/CACAO as distributed by the MT, apply the LGPL giveback clause as well and publish said modifications under LGPL.
